"""Skip DeepSeek V4 Indexer scoring when its top-k would select every row.

The Indexer picks ``index_topk`` compressed rows for each query. While the
compressed pool holds no more than ``index_topk`` rows (contexts shorter than
``index_topk * compress_ratio`` tokens) that selection is every row, so the
scoring (wq_b, weights_proj, einsum, argpartition) and the attention-side
gather only produce a permutation of the full pool, and attention does not
depend on key order. Returning ``None`` sends V4Attention down its
full-pool path, which attends to exactly the same keys.

Only uniform-batch steps of up to a few tokens (decode and speculative
verification) take the shortcut.
The Indexer's own compressor still runs every step so its pool is ready once
the context outgrows ``index_topk``.

Longer uniform steps (prefill chunks) score with mlx_lm's formula too, but
fold the ReLU and the weighted sum over heads into one kernel: the
``[S, n_heads, pool]`` score tensor is read once instead of being rewritten
by the ReLU and read again by the head reduction.
"""

import math

import mlx.core as mx
from mlx_lm.models.deepseek_v4 import (
    _K_IDX,  # pyright: ignore[reportPrivateUsage]
    DeepseekV4Cache,
    Indexer,
)

_original_call = Indexer.__call__
# Decode and short speculative-verification steps.
_MAX_SHORTCUT_SEQUENCE = 4
_THREADGROUP = 256

# out[row, t] = sum_h weights[row, h] * max(scores[row, h, t], 0), accumulated
# in float32 like the matmul it replaces.
_RELU_WEIGHTED_SUM_SOURCE = """
    uint t = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;
    if (t >= POOL) {
        return;
    }
    const device T_IN* row_scores = scores + row * HEADS * POOL + t;
    const device T_IN* row_weights = weights + row * HEADS;
    float total = 0.0f;
    for (uint h = 0; h < HEADS; ++h) {
        float score = static_cast<float>(row_scores[h * POOL]);
        total += static_cast<float>(row_weights[h]) * metal::max(score, 0.0f);
    }
    out[row * POOL + t] = static_cast<T_IN>(total);
"""

_relu_weighted_sum_kernel = mx.fast.metal_kernel(
    name="dsv4_indexer_relu_weighted_sum",
    input_names=["scores", "weights"],
    output_names=["out"],
    source=_RELU_WEIGHTED_SUM_SOURCE,
)


def relu_weighted_sum(scores: mx.array, weights: mx.array) -> mx.array:
    """``(weights[..., None, :] @ relu(scores)).squeeze(-2)`` for scores
    ``[B, S, H, T]`` and weights ``[B, S, H]``; returns ``[B, S, T]``."""
    batch, sequence, heads, pool = scores.shape
    return _relu_weighted_sum_kernel(
        inputs=[scores, weights.astype(scores.dtype)],
        template=[("HEADS", heads), ("POOL", pool), ("T_IN", scores.dtype)],
        grid=(pool, batch * sequence, 1),
        threadgroup=(_THREADGROUP, 1, 1),
        output_shapes=[(batch, sequence, pool)],
        output_dtypes=[scores.dtype],
    )[0]


def _patched_call(
    self: Indexer,
    x: mx.array,
    qr: mx.array,
    cache: DeepseekV4Cache,
    offset: int | mx.array,
) -> mx.array | None:
    prefill = x.shape[1] > _MAX_SHORTCUT_SEQUENCE
    if cache.pooled_lengths(_K_IDX) is not None or (
        prefill and isinstance(offset, mx.array)
    ):
        return _original_call(self, x, qr, cache, offset)

    idx_kv = self.compressor(x, cache, offset, key=_K_IDX)
    if prefill:
        if idx_kv.shape[1] == 0:
            return None
        return _score_and_select(self, x, qr, idx_kv, offset, fused=True)
    if idx_kv.shape[1] <= self.index_topk:
        return None
    return _score_and_select(self, x, qr, idx_kv, offset)


def _score_and_select(
    self: Indexer,
    x: mx.array,
    qr: mx.array,
    idx_kv: mx.array,
    offset: int | mx.array,
    fused: bool = False,
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
    if fused:
        score = relu_weighted_sum(score, per_head_weights)
    else:
        score = mx.maximum(score, 0)
        score = mx.matmul(per_head_weights[:, :, None, :], score).squeeze(2)
    k = min(self.index_topk, idx_kv.shape[1])
    return mx.argpartition(-score, kth=k - 1, axis=-1)[..., :k].astype(mx.int32)


def patch_deepseek_v4_indexer() -> None:
    Indexer.__call__ = _patched_call
