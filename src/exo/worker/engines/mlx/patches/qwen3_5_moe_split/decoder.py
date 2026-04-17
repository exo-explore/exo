"""Attention/MoE split DecoderLayer.__call__ for Qwen3.5.

Two modes:
  S == 1 (decode): serial split with all_gather + dummy layernorm on idle rank.
  S > 1 (prefill/verify): within-layer pipelined split — ATTN does H0 then H1,
    MOE does H0 while ATTN does H1 (overlap), then MOE does H1.
"""

from collections.abc import Callable

import mlx.core as mx

ATTN_RANK = 0
MOE_RANK = 1


def make_split_decoder_call(
    group: mx.distributed.Group,
    n_layers: int = 48,
) -> Callable[..., mx.array]:
    """Build a DecoderLayer.__call__ replacement closed over ``group``."""
    if group.size() != 2:
        raise ValueError(
            f"qwen3_5_moe_split requires world_size==2, got {group.size()}"
        )
    rank = group.rank()
    _layer_count = 0
    _cached_masks: dict[int, tuple] = {}
    # Cross-layer pipeline state: previous layer's self + pending h_H1
    _prev_layer_self = None
    _pending_h_H1: mx.array | None = None

    def _attn(self, x, mask, cache):  # type: ignore[no-untyped-def]
        if self.is_linear:
            r = self.linear_attn(self.input_layernorm(x), mask, cache)
        else:
            r = self.self_attn(self.input_layernorm(x), mask, cache)
        return x + r

    def _get_masks(S, mid, mask, cache):  # type: ignore[no-untyped-def]
        if S in _cached_masks:
            return _cached_masks[S]
        if mask is None:
            result = (None, None)
        else:
            if hasattr(cache, 'offset'):
                offset = cache.offset
            elif hasattr(cache, 'cache') and cache.cache[0] is not None:
                offset = cache.cache[0].shape[1]
            else:
                offset = 0
            result = (mask[:, :, :mid, :offset + mid], mask[:, :, mid:, :])
        _cached_masks[S] = result
        return result

    def _split_call(self, x, mask=None, cache=None):  # type: ignore[no-untyped-def]
        nonlocal _layer_count, _prev_layer_self, _pending_h_H1
        _layer_count += 1
        S = x.shape[1]

        # ---- S == 1: serial decode (Phase 1a path) ----
        if S == 1:
            _cached_masks.clear()
            _prev_layer_self = None
            _pending_h_H1 = None
            if rank == ATTN_RANK:
                h = _attn(self, x, mask, cache)
            else:
                h = x
                for _ in range(50):
                    h = self.post_attention_layernorm(h) + h
                if _layer_count % 2 == 0:
                    mx.eval(h)
            h = mx.distributed.all_gather(h, group=group)[ATTN_RANK : ATTN_RANK + 1]

            if rank == MOE_RANK:
                out = h + self.mlp(self.post_attention_layernorm(h))
            else:
                out = h
                for _ in range(50):
                    out = self.input_layernorm(out) + out
                if _layer_count % 2 == 0:
                    mx.eval(out)
            out = mx.distributed.all_gather(out, group=group)[MOE_RANK : MOE_RANK + 1]
            return out

        # ---- S > 1: cross-layer pipelined prefill/verify ----
        mid = S // 2
        x_H0 = x[:, :mid, :]
        x_H1 = x[:, mid:, :]
        mask_H0, mask_H1 = _get_masks(S, mid, mask, cache)
        is_last_layer = (_layer_count % n_layers == 0)

        # Stage A: ATTN does H0 attention || MOE does prev layer's H1 MoE
        if rank == ATTN_RANK:
            h_H0 = _attn(self, x_H0, mask_H0, cache)
        else:
            h_H0 = x_H0  # placeholder

        if rank == MOE_RANK and _pending_h_H1 is not None:
            # Finish previous layer's H1 using previous layer's MLP
            out_H1_prev = _pending_h_H1 + _prev_layer_self.mlp(
                _prev_layer_self.post_attention_layernorm(_pending_h_H1)
            )
        else:
            out_H1_prev = None

        h_H0 = mx.distributed.all_gather(h_H0, group=group)[ATTN_RANK : ATTN_RANK + 1]
        if out_H1_prev is not None:
            out_H1_prev = mx.distributed.all_gather(out_H1_prev, group=group)[MOE_RANK : MOE_RANK + 1]
        mx.eval(h_H0)

        # Stage B: ATTN does H1 attention || MOE does this layer's H0 MoE
        if rank == ATTN_RANK:
            h_H1 = _attn(self, x_H1, mask_H1, cache)
        else:
            h_H1 = x_H1  # placeholder

        if rank == MOE_RANK:
            out_H0 = h_H0 + self.mlp(self.post_attention_layernorm(h_H0))
        else:
            out_H0 = h_H0  # placeholder

        h_H1 = mx.distributed.all_gather(h_H1, group=group)[ATTN_RANK : ATTN_RANK + 1]
        out_H0 = mx.distributed.all_gather(out_H0, group=group)[MOE_RANK : MOE_RANK + 1]
        mx.eval(h_H1)
        mx.eval(out_H0)

        # Save state for next layer's Stage A
        _prev_layer_self = self
        _pending_h_H1 = h_H1

        # Determine out_H1 for the return value
        if is_last_layer:
            # Last layer: flush — MOE processes H1 now
            if rank == MOE_RANK:
                out_H1 = h_H1 + self.mlp(self.post_attention_layernorm(h_H1))
            else:
                out_H1 = h_H1
            out_H1 = mx.distributed.all_gather(out_H1, group=group)[MOE_RANK : MOE_RANK + 1]
            mx.eval(out_H1)
            _prev_layer_self = None
            _pending_h_H1 = None
        elif out_H1_prev is not None:
            # Layers 1+: use prev layer's H1 MoE result (computed in Stage A)
            mx.eval(out_H1_prev)
            out_H1 = out_H1_prev
        else:
            # Layer 0: startup bubble — MOE processes H1 now (no overlap)
            if rank == MOE_RANK:
                out_H1 = h_H1 + self.mlp(self.post_attention_layernorm(h_H1))
            else:
                out_H1 = h_H1
            out_H1 = mx.distributed.all_gather(out_H1, group=group)[MOE_RANK : MOE_RANK + 1]
            mx.eval(out_H1)

        return mx.concatenate([out_H0, out_H1], axis=1)

    return _split_call
