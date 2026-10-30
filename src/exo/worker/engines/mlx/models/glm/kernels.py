# type: ignore
# basedpyright skips this file (mx.fast.metal_kernel is not in the repo's mlx stubs);
# kernels.pyi types it for the rest of exo.
"""Metal kernels for GLM-5.x (DeepSeek sparse attention) prefill.

Both are used only where mx.fast.metal_kernel can run (a Metal GPU as the default device);
glm_moe_dsa.py falls back to mlx-lm's code elsewhere.
"""

import mlx.core as mx

_INDEXER_SCORES_SOURCE = r"""
    const uint n = thread_position_in_grid.x;
    const uint l = thread_position_in_grid.y;
    const uint b = thread_position_in_grid.z;
    const uint H = s_shape[1];
    const uint L = s_shape[2];
    const uint N = s_shape[3];
    if (n >= N) return;
    float acc = -INFINITY;
    bool keep = true;
    if (HAS_MASK) {
        keep = mask[b * mask_strides[0] + l * mask_strides[2] + n * mask_strides[3]];
    }
    if (keep) {
        acc = 0.0f;
        for (uint h = 0; h < H; h++) {
            const float v = float(
                s[b * s_strides[0] + h * s_strides[1] + l * s_strides[2] + n * s_strides[3]]);
            acc += metal::max(v, 0.0f) * float(w[b * w_strides[0] + h * w_strides[1] + l * w_strides[2]]);
        }
    }
    out[(size_t(b) * L + l) * N + n] = acc;
"""

_GATHER_ROWS_SOURCE = r"""
    const uint c = thread_position_in_grid.x;
    const uint r = thread_position_in_grid.y;
    const uint C = (src_shape[1] * sizeof(T)) / 16;
    if (c >= C) return;
    const uint row = uint(idx[r]);
    const device uint4* s = (const device uint4*)(src) + size_t(row) * C;
    device uint4* o = (device uint4*)(out) + size_t(r) * C;
    o[c] = s[c];
"""

_indexer_scores_kernel = None
_gather_rows_kernel = None


def metal_kernels_available() -> bool:
    return mx.metal.is_available() and mx.default_device() == mx.gpu


def indexer_scores(
    scores: mx.array, weights: mx.array, mask: mx.array | None
) -> mx.array:
    """The DSA indexer's per-key score: sum over its heads of relu(q.k) * w, in float32.

    scores [B, H, L, N] holds the heads' q.k products, weights [B, H, L, 1] their weights,
    mask (boolean, broadcastable to [B, 1, L, N]) the keys each query may select. Returns
    float32 [B, 1, L, N], -inf where masked.

    mlx-lm adds the heads one at a time in bfloat16 (31 rounded additions of [B, 1, L, N]
    tensors), which is slow and, with only a few hundred distinct values per row, makes many
    keys tie at the top-k boundary. Summing in float32 in one pass is exact to float32
    rounding and selects as transformers' float32 indexer does.
    """
    global _indexer_scores_kernel
    if _indexer_scores_kernel is None:
        _indexer_scores_kernel = mx.fast.metal_kernel(
            name="glm_dsa_indexer_scores",
            input_names=["s", "w", "mask"],
            output_names=["out"],
            source=_INDEXER_SCORES_SOURCE,
            ensure_row_contiguous=False,
        )
    batch, _, length, n_keys = scores.shape
    has_mask = mask is not None
    if mask is not None:
        if mask.dtype != mx.bool_:
            raise ValueError("indexer_scores expects a boolean mask")
        mask = mx.broadcast_to(
            mask.reshape((1,) * (4 - mask.ndim) + mask.shape),
            (batch, 1, length, n_keys),
        )
    else:
        mask = mx.zeros((1, 1, 1, 1), mx.bool_)
    outputs = _indexer_scores_kernel(
        inputs=[scores, weights, mask],
        template=[("HAS_MASK", has_mask)],
        grid=(n_keys, length, batch),
        threadgroup=(min(256, n_keys), 1, 1),
        output_shapes=[(batch, 1, length, n_keys)],
        output_dtypes=[mx.float32],
    )
    return outputs[0]


def gather_rows(source: mx.array, indices: mx.array) -> mx.array:
    """source[indices] for a row-contiguous [N, D] source whose rows are a multiple of 16
    bytes, copied with 16-byte vector loads. Returns indices.shape + (D,).

    The same rows as mx.take, about three times faster for the [queries, 2048, 576] key
    gather of sparse prefill."""
    global _gather_rows_kernel
    if _gather_rows_kernel is None:
        _gather_rows_kernel = mx.fast.metal_kernel(
            name="glm_dsa_gather_rows",
            input_names=["src", "idx"],
            output_names=["out"],
            source=_GATHER_ROWS_SOURCE,
        )
    width = source.shape[-1]
    if source.ndim != 2 or (width * source.itemsize) % 16 != 0:
        raise ValueError("gather_rows needs [N, D] rows of a multiple of 16 bytes")
    chunks = width * source.itemsize // 16
    n_rows = indices.size
    outputs = _gather_rows_kernel(
        inputs=[source, indices.reshape(-1)],
        template=[("T", source.dtype)],
        grid=(chunks, n_rows, 1),
        threadgroup=(chunks, max(1, 256 // chunks), 1),
        output_shapes=[(n_rows, width)],
        output_dtypes=[source.dtype],
    )
    return outputs[0].reshape(*indices.shape, width)
