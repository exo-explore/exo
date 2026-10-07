"""Skip DeepSeek V4 Indexer scoring when its top-k would select every row.

The Indexer picks ``index_topk`` compressed rows for each query. While the
compressed pool holds no more than ``index_topk`` rows (contexts shorter than
``index_topk * compress_ratio`` tokens) that selection is every row, so the
scoring (wq_b, weights_proj, einsum, argpartition) and the attention-side
gather only produce a permutation of the full pool, and attention does not
depend on key order. Returning ``None`` sends V4Attention down its
full-pool path, which attends to exactly the same keys.

Only the single-token decode step with a uniform batch takes the shortcut.
The Indexer's own compressor still runs every step so its pool is ready once
the context outgrows ``index_topk``.
"""

import math

import mlx.core as mx
from mlx_lm.models.deepseek_v4 import (
    _K_IDX,  # pyright: ignore[reportPrivateUsage]
    DeepseekV4Cache,
    Indexer,
)

_original_call = Indexer.__call__


def _patched_call(
    self: Indexer,
    x: mx.array,
    qr: mx.array,
    cache: DeepseekV4Cache,
    offset: int | mx.array,
) -> mx.array | None:
    if x.shape[1] != 1 or cache.pooled_lengths(_K_IDX) is not None:
        return _original_call(self, x, qr, cache, offset)

    idx_kv = self.compressor(x, cache, offset, key=_K_IDX)
    if idx_kv.shape[1] <= self.index_topk:
        return None
    return _score_and_select(self, x, qr, idx_kv, offset)


def _score_and_select(
    self: Indexer,
    x: mx.array,
    qr: mx.array,
    idx_kv: mx.array,
    offset: int | mx.array,
) -> mx.array:
    """The scoring half of ``Indexer.__call__`` for a uniform batch."""
    batch, sequence, _ = x.shape
    rope_dim = self.rope_head_dim
    q = self.wq_b(qr).reshape(batch, sequence, self.n_heads, self.head_dim)
    q = mx.concatenate(
        [q[..., :-rope_dim], self.rope(q[..., -rope_dim:], offset=offset)], axis=-1
    )
    per_head_weights = mx.multiply(
        self.weights_proj(x), self.softmax_scale * math.pow(self.n_heads, -0.5)
    )
    score = mx.einsum(  # pyright: ignore[reportUnknownMemberType]
        "bshd,btd->bsht", q.astype(idx_kv.dtype), idx_kv
    )
    score = mx.maximum(score, 0)
    score = mx.matmul(per_head_weights[:, :, None, :], score).squeeze(2)
    k = min(self.index_topk, idx_kv.shape[1])
    return mx.argpartition(-score, kth=k - 1, axis=-1)[..., :k].astype(mx.int32)


def patch_deepseek_v4_indexer() -> None:
    Indexer.__call__ = _patched_call
