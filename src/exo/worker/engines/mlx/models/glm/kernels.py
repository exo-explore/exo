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

_ROUTER_SELECT_SOURCE = r"""
    // One threadgroup per token, one thread per expert (E <= 1024, a multiple of 32).
    const uint t = threadgroup_position_in_grid.x;
    const uint e = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    threadgroup float group_best[32];
    threadgroup uint group_index[32];
    threadgroup float chosen_score[K];
    threadgroup uint chosen_index[K];
    const float s = 1.0f / (1.0f + metal::exp(-logits[size_t(t) * E + e]));
    float biased = s + bias[e];
    for (int k = 0; k < K; k++) {
        float v = biased;
        uint i = e;
        for (uint offset = 16; offset > 0; offset >>= 1) {
            const float other_v = simd_shuffle_down(v, offset);
            const uint other_i = simd_shuffle_down(i, offset);
            if (other_v > v || (other_v == v && other_i < i)) {
                v = other_v;
                i = other_i;
            }
        }
        if (lane == 0) {
            group_best[sg] = v;
            group_index[sg] = i;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (e == 0) {
            float best = group_best[0];
            uint best_i = group_index[0];
            for (uint g = 1; g < E / 32; g++) {
                if (group_best[g] > best || (group_best[g] == best && group_index[g] < best_i)) {
                    best = group_best[g];
                    best_i = group_index[g];
                }
            }
            chosen_index[k] = best_i;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (e == chosen_index[k]) {
            chosen_score[k] = s;
            biased = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (e == 0) {
        float denominator = 0.0f;
        for (int k = 0; k < K; k++) denominator += chosen_score[k];
        for (int k = 0; k < K; k++) {
            float weight = chosen_score[k];
            if (NORMALIZE) weight = weight / denominator;
            inds[size_t(t) * K + k] = chosen_index[k];
            scores[size_t(t) * K + k] = weight * scaling[0];
        }
    }
"""

_MOE_COMBINE_SOURCE = r"""
    // routed [T, K, D], weights [T, K] float32, shared [T, D] -> out [T, D]
    const uint d = thread_position_in_grid.x;
    const uint t = thread_position_in_grid.y;
    if (d >= D) return;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        acc += float(routed[(size_t(t) * K + k) * D + d]) * weights[size_t(t) * K + k];
    }
    const T routed_sum = T(acc);
    out[size_t(t) * D + d] = T(float(routed_sum) + float(shared[size_t(t) * D + d]));
"""

_indexer_scores_kernel = None
_gather_rows_kernel = None
_router_select_kernel = None
_moe_combine_kernel = None


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


def router_select(
    logits: mx.array,
    bias: mx.array,
    top_k: int,
    routed_scaling_factor: float,
    normalize: bool,
) -> tuple[mx.array, mx.array]:
    """mlx-lm's group_expert_select for one expert group (n_group == 1), in one kernel:
    sigmoid scores, the bias added only to choose the top_k experts, the chosen experts'
    unbiased scores normalised (if normalize) and scaled. Ties go to the lower expert
    index. logits [..., E] float32 -> (indices uint32 [..., top_k], scores float32)."""
    global _router_select_kernel
    if _router_select_kernel is None:
        _router_select_kernel = mx.fast.metal_kernel(
            name="glm_router_select",
            input_names=["logits", "bias", "scaling"],
            output_names=["inds", "scores"],
            source=_ROUTER_SELECT_SOURCE,
        )
    *lead, n_experts = logits.shape
    if n_experts % 32 != 0 or n_experts > 1024:
        raise ValueError("router_select needs a multiple of 32 experts, at most 1024")
    tokens = 1
    for size in lead:
        tokens *= size
    inds, scores = _router_select_kernel(
        inputs=[
            logits.reshape(tokens, n_experts),
            bias.astype(mx.float32),
            mx.array([routed_scaling_factor], mx.float32),
        ],
        template=[("E", n_experts), ("K", top_k), ("NORMALIZE", normalize)],
        grid=(tokens * n_experts, 1, 1),
        threadgroup=(n_experts, 1, 1),
        output_shapes=[(tokens, top_k), (tokens, top_k)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return inds.reshape(*lead, top_k), scores.reshape(*lead, top_k)


def moe_combine(routed: mx.array, weights: mx.array, shared: mx.array) -> mx.array:
    """(routed * weights[..., None]).sum(-2) in float32, rounded to routed's dtype, plus
    the shared expert's output: mlx-lm's MoE combine (three kernels) as one, same rounding.
    routed [..., K, D], weights [..., K] float32, shared [..., D] -> [..., D]."""
    global _moe_combine_kernel
    if _moe_combine_kernel is None:
        _moe_combine_kernel = mx.fast.metal_kernel(
            name="glm_moe_combine",
            input_names=["routed", "weights", "shared"],
            output_names=["out"],
            source=_MOE_COMBINE_SOURCE,
        )
    *lead, n_chosen, width = routed.shape
    tokens = 1
    for size in lead:
        tokens *= size
    (out,) = _moe_combine_kernel(
        inputs=[
            routed.reshape(tokens, n_chosen, width),
            weights.reshape(tokens, n_chosen),
            shared.reshape(tokens, width),
        ],
        template=[("T", routed.dtype), ("K", n_chosen), ("D", width)],
        grid=(width, tokens, 1),
        threadgroup=(min(256, width), 1, 1),
        output_shapes=[(tokens, width)],
        output_dtypes=[routed.dtype],
    )
    return out.reshape(*lead, width)
