"""Attention/MoE split DecoderLayer.__call__ for Qwen3.5 MoE.

Two-rank split: rank 0 runs input_layernorm + attention + first residual,
rank 1 runs post_attention_layernorm + MoE + second residual. One cross-rank
send/recv pair per layer.

Graph-splitting idiom borrowed from auto_parallel.PipelineFirstLayer /
PipelineLastLayer: blocking mx.eval before and after every distributed op so
the send/recv stays on its own Metal command buffer (avoids GPU timeout).

No mx.depends cache anchoring in v1 — plain mx.eval only. If a send gets
pruned from the lazy graph we'll add mx.depends(cache.keys, h) as a targeted
fix, mirroring PipelineLastLayer:208-216.
"""

from collections.abc import Callable

import mlx.core as mx

ATTN_RANK = 0
MOE_RANK = 1


def make_split_decoder_call(
    group: mx.distributed.Group,
) -> Callable[..., mx.array]:
    """Build a DecoderLayer.__call__ replacement closed over ``group``.

    Raises ValueError if group.size() != 2.
    """
    if group.size() != 2:
        raise ValueError(
            f"qwen3_5_moe_split requires world_size==2, got {group.size()}"
        )
    rank = group.rank()

    def _split_call(self, x, mask=None, cache=None):  # type: ignore[no-untyped-def]
        if rank == ATTN_RANK:
            if self.is_linear:
                r = self.linear_attn(self.input_layernorm(x), mask, cache)
            else:
                r = self.self_attn(self.input_layernorm(x), mask, cache)
            h = x + r
            h = mx.distributed.send(h, MOE_RANK, group=group)
            mx.eval(h)
            out = mx.distributed.recv_like(h, MOE_RANK, group=group)
            return out

        h = mx.distributed.recv_like(x, ATTN_RANK, group=group)
        out = h + self.mlp(self.post_attention_layernorm(h))
        sent = mx.distributed.send(out, ATTN_RANK, group=group)
        mx.eval(sent)
        return out

    return _split_call
