"""Serial attn/MoE split DecoderLayer.__call__ — S == 1 decode only.

For S > 1 (prefill / speculative verify), the model-level pipelined
forward in `model_forward.py` calls the layer primitives directly and
never invokes this function.

S == 1 path: ATTN rank runs attention + residual, MOE rank runs MoE +
residual. Each step ends in one all_gather; the idle rank does dummy
layernorm work, evaled every ~2 layers to prevent JACCL queue overflow.
"""

from collections.abc import Callable

import mlx.core as mx

ATTN_RANK = 0
MOE_RANK = 1
DUMMY_LN_ITERS = 50


def make_split_decoder_call(
    group: mx.distributed.Group,
    n_layers: int = 48,
) -> Callable[..., mx.array]:
    """Build a DecoderLayer.__call__ replacement for S == 1 decode."""
    if group.size() != 2:
        raise ValueError(f"world_size==2 required, got {group.size()}")
    rank = group.rank()
    layer_counter = 0

    def gather_from(rank_to_pick: int, tensor: mx.array) -> mx.array:
        """all_gather and slice the tensor contributed by ``rank_to_pick``."""
        gathered = mx.distributed.all_gather(tensor, group=group)
        return gathered[rank_to_pick : rank_to_pick + 1]

    def _split_call(self, x, mask=None, cache=None):  # type: ignore[no-untyped-def]
        nonlocal layer_counter
        layer_counter += 1
        lc = layer_counter

        # Step 1: ATTN does attention + residual; MOE does dummy layernorm.
        if rank == ATTN_RANK:
            if self.is_linear:
                r = self.linear_attn(self.input_layernorm(x), mask, cache)
            else:
                r = self.self_attn(self.input_layernorm(x), mask, cache)
            h = x + r
        else:
            h = x
            for _ in range(DUMMY_LN_ITERS):
                h = self.post_attention_layernorm(h) + h
            if lc % 2 == 0:
                mx.eval(h)
        h = gather_from(ATTN_RANK, h)
        mx.eval(h)
        print(f"[rank {rank}] L={lc} after gather-1  h.mean={h.mean().item():+.6f}", flush=True)

        # Step 2: MOE does MoE + residual; ATTN does dummy layernorm.
        if rank == MOE_RANK:
            out = h + self.mlp(self.post_attention_layernorm(h))
        else:
            out = h
            for _ in range(DUMMY_LN_ITERS):
                out = self.input_layernorm(out) + out
            if lc % 2 == 0:
                mx.eval(out)
        result = gather_from(MOE_RANK, out)
        mx.eval(result)
        print(f"[rank {rank}] L={lc} after gather-2 out.mean={result.mean().item():+.6f}", flush=True)
        return result

    return _split_call
