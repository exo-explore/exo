"""Attention/MoE split DecoderLayer.__call__ for Qwen3.5.

Two modes keyed on sequence length S:

  S == 1 (decode): serial 2-step per layer. ATTN does attention, all_gather
    broadcasts h to both ranks. MOE does MoE, all_gather broadcasts out.
    The idle rank does dummy layernorm work so both ranks' graphs are
    non-trivial; eval every ~2 layers prevents JACCL queue overflow.

  S > 1 (prefill / speculative verify): cross-layer pipelined 2-stage per
    layer with overlap. Sequence is cut in half — H0 = first S/2 tokens,
    H1 = last S/2 tokens.

    Stage A: ATTN computes h_H0 (attention on H0).
             MOE simultaneously finishes H1 from the PREVIOUS layer
             (i.e. MoE on pending h_H1 using previous layer's MLP).
             Both ranks do real work in parallel (the overlap).

    Stage B: ATTN computes h_H1 (attention on H1, using H0's KV cache
             from Stage A).
             MOE simultaneously runs MoE on h_H0 (this layer's MLP).
             Both ranks do real work in parallel (the overlap).

    Layer 0 has no previous-layer pending H1, so Stage A's MOE is idle
    (startup bubble). Layer N-1 flushes the pending H1 at the end so the
    full pipeline drains (drain bubble). Middle layers get full overlap.
"""

from collections.abc import Callable
from typing import Any

import mlx.core as mx

ATTN_RANK = 0
MOE_RANK = 1
DUMMY_LN_ITERS = 50  # S==1 dummy work count


def make_split_decoder_call(
    group: mx.distributed.Group,
    n_layers: int = 48,
) -> Callable[..., mx.array]:
    """Build a DecoderLayer.__call__ replacement closed over ``group``."""
    if group.size() != 2:
        raise ValueError(f"world_size==2 required, got {group.size()}")
    rank = group.rank()

    # Per-forward-pass state (reset on every S==1 decode call)
    state: dict[str, Any] = {
        "layer_idx": 0,          # 1-indexed; 1 == first layer of current forward
        "mask_cache": {},        # {S: (mask_H0, mask_H1)}
        "prev_layer": None,      # previous layer's `self` (for its MLP)
        "pending_h_H1": None,    # previous layer's h_H1 (awaiting MoE)
    }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def attention(self, x, mask, cache):  # type: ignore[no-untyped-def]
        """Layer's attention (GDN or GQA) + residual. Returns h = x + attn(x)."""
        if self.is_linear:
            r = self.linear_attn(self.input_layernorm(x), mask, cache)
        else:
            r = self.self_attn(self.input_layernorm(x), mask, cache)
        return x + r

    def moe(self, h):  # type: ignore[no-untyped-def]
        """Layer's MoE + residual. Returns out = h + mlp(ln(h))."""
        return h + self.mlp(self.post_attention_layernorm(h))

    def gather_from(rank_to_pick: int, tensor: mx.array) -> mx.array:
        """all_gather and pick the slice contributed by `rank_to_pick`."""
        gathered = mx.distributed.all_gather(tensor, group=group)
        return gathered[rank_to_pick : rank_to_pick + 1]

    def get_masks(S: int, mid: int, mask, cache):  # type: ignore[no-untyped-def]
        """Slice the full mask into H0 and H1 masks. Cached by S."""
        cached = state["mask_cache"].get(S)
        if cached is not None:
            return cached
        if mask is None:
            result = (None, None)
        else:
            # Offset = KV length BEFORE this forward pass. Needed so H0's
            # mask columns match H0's attention's actual KV length
            # (offset prior + mid new from H0).
            if hasattr(cache, "offset"):
                offset = cache.offset
            elif hasattr(cache, "cache") and cache.cache[0] is not None:
                offset = cache.cache[0].shape[1]
            else:
                offset = 0
            # H0: queries [0, mid), keys [0, offset+mid)
            # H1: queries [mid, S), keys [0, offset+S) — full width since
            #     after H0 is cached the KV has offset+S entries.
            result = (mask[:, :, :mid, : offset + mid], mask[:, :, mid:, :])
        state["mask_cache"][S] = result
        return result

    # ------------------------------------------------------------------
    # S == 1: serial decode
    # ------------------------------------------------------------------

    def serial_decode(self, x, mask, cache, layer_idx):  # type: ignore[no-untyped-def]
        # Step 1: ATTN does attention; MOE does dummy layernorm.
        if rank == ATTN_RANK:
            h = attention(self, x, mask, cache)
        else:
            h = x
            for _ in range(DUMMY_LN_ITERS):
                h = self.post_attention_layernorm(h) + h
            if layer_idx % 2 == 0:
                mx.eval(h)  # drain JACCL queue periodically
        h = gather_from(ATTN_RANK, h)

        # Step 2: MOE does MoE; ATTN does dummy layernorm.
        if rank == MOE_RANK:
            out = moe(self, h)
        else:
            out = h
            for _ in range(DUMMY_LN_ITERS):
                out = self.input_layernorm(out) + out
            if layer_idx % 2 == 0:
                mx.eval(out)
        return gather_from(MOE_RANK, out)

    # ------------------------------------------------------------------
    # S > 1: cross-layer pipelined prefill / verify
    # ------------------------------------------------------------------

    def pipelined_forward(self, x, mask, cache, layer_idx):  # type: ignore[no-untyped-def]
        S = x.shape[1]
        mid = S // 2
        x_H0 = x[:, :mid, :]
        x_H1 = x[:, mid:, :]
        mask_H0, mask_H1 = get_masks(S, mid, mask, cache)
        is_first_layer = state["pending_h_H1"] is None
        is_last_layer = layer_idx % n_layers == 0

        # -------- Stage A --------
        # ATTN: attention on H0
        # MOE:  MoE on prev-layer's h_H1 (startup bubble on layer 0)
        if rank == ATTN_RANK:
            h_H0 = attention(self, x_H0, mask_H0, cache)
        else:
            h_H0 = x_H0  # placeholder (will be discarded by all_gather slice)

        if not is_first_layer:
            prev_pending = state["pending_h_H1"]
            if rank == MOE_RANK:
                prev = state["prev_layer"]
                # Critical: use PREVIOUS layer's MLP weights, not self.mlp
                out_H1_prev = prev_pending + prev.mlp(
                    prev.post_attention_layernorm(prev_pending)
                )
            else:
                out_H1_prev = prev_pending  # placeholder on ATTN_RANK
        else:
            out_H1_prev = None

        h_H0 = gather_from(ATTN_RANK, h_H0)
        if out_H1_prev is not None:
            out_H1_prev = gather_from(MOE_RANK, out_H1_prev)
        mx.eval(h_H0)

        # -------- Stage B --------
        # ATTN: attention on H1 (H0's KV now in cache)
        # MOE:  MoE on h_H0 (this layer's MLP)
        if rank == ATTN_RANK:
            h_H1 = attention(self, x_H1, mask_H1, cache)
        else:
            h_H1 = x_H1  # placeholder

        if rank == MOE_RANK:
            out_H0 = moe(self, h_H0)
        else:
            out_H0 = h_H0  # placeholder

        h_H1 = gather_from(ATTN_RANK, h_H1)
        out_H0 = gather_from(MOE_RANK, out_H0)
        mx.eval(h_H1)
        mx.eval(out_H0)

        # Save state so the NEXT layer's Stage A can finish this layer's H1
        state["prev_layer"] = self
        state["pending_h_H1"] = h_H1

        # -------- Determine out_H1 for the return --------
        if is_first_layer:
            # Layer 0: no prior H1 pipelined, so MoE H1 serially (bubble)
            if rank == MOE_RANK:
                out_H1 = moe(self, h_H1)
            else:
                out_H1 = h_H1
            out_H1 = gather_from(MOE_RANK, out_H1)
            mx.eval(out_H1)
        elif is_last_layer:
            # Layer N-1: flush pending H1 immediately (bubble)
            if rank == MOE_RANK:
                out_H1 = moe(self, h_H1)
            else:
                out_H1 = h_H1
            out_H1 = gather_from(MOE_RANK, out_H1)
            mx.eval(out_H1)
            state["prev_layer"] = None
            state["pending_h_H1"] = None
        else:
            # Middle layer: out_H1 comes from Stage A's prev-layer MoE (free)
            mx.eval(out_H1_prev)
            out_H1 = out_H1_prev

        return mx.concatenate([out_H0, out_H1], axis=1)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _split_call(self, x, mask=None, cache=None):  # type: ignore[no-untyped-def]
        state["layer_idx"] += 1
        layer_idx = state["layer_idx"]
        S = x.shape[1]

        if S == 1:
            # Reset pipeline state on decode (new forward pass boundary)
            state["mask_cache"].clear()
            state["prev_layer"] = None
            state["pending_h_H1"] = None
            return serial_decode(self, x, mask, cache, layer_idx)

        return pipelined_forward(self, x, mask, cache, layer_idx)

    return _split_call
