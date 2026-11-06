# type: ignore
# Typed by minimax_m3.pyi next to this file (MLX's mx.fast has no stubs here).
"""MiniMax M3 / M3.1 (``minimax_m3_vl``): mlx-lm's model with a faster decode step.

The classes subclass mlx-lm's (``mlx_lm/models/minimax_m3_vl.py`` at the pinned revision)
and keep its weights, parameter names, sanitizing, caches and sharding. What changes:

- **Gemma-style norms** apply ``1 + weight``. mlx-lm adds the 1 at every call; here it is
  added once, on the first call, in the weight's dtype (bitwise the same).
- **Sparse decode reads only the selected blocks.** For a one-token step past the block
  budget, mlx-lm cast the whole index-key cache to float32, scored every key, built a
  boolean mask over every head and every cached key, and ran dense attention over all of
  them: about 20 dependent operations per layer. Three Metal kernels now do it: one
  scores the keys in float32 straight from the bf16 cache and keeps each block's best
  score; one, per KV group and selected block, picks the same blocks by mlx-lm's rule
  (the 16 best, the query's own block always, ties to the lower block) and attends to
  that block's keys for every query head of the group; one merges the blocks' partial
  softmaxes. The same softmax over the same keys, summed in another order.
- **Sparse prefill** applies the selection per KV group, as one attention call with the
  groups in the batch axis, instead of repeating the mask for all 64 query heads (bitwise
  the same; 16x less mask memory).
- **Routing** (sigmoid, bias, top-4, normalization, cast) is one Metal kernel per token
  instead of about eight operations, bitwise mlx-lm's: the same formulas, the choice in
  the order mlx's argpartition (a stable sort) gives, the chosen scores summed in that
  order. Other shapes and dtypes use mlx-lm's ops.
- **Expert combine:** the weighted sum of the routed experts' outputs plus the shared
  expert's runs as one compiled kernel instead of three, adding in the order MLX's
  reduction over the expert axis does (bitwise in bf16).

Everything but the sparse decode step is bitwise mlx-lm's; the tests check it at every
prefill length across MLX's kernel thresholds.
"""

from collections.abc import Callable
from functools import cache, partial
from typing import Any, cast, override

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import minimax_m3_vl as upstream
from mlx_lm.models.activations import SwigluOAI
from mlx_lm.models.base import scaled_dot_product_attention


class GemmaRMSNorm(upstream.GemmaRMSNorm):
    """RMS norm with weight ``1 + w``, the sum computed once (in ``w``'s dtype)."""

    @override
    def __call__(self, x: mx.array) -> mx.array:
        scale = cast(mx.array | None, self.get("_scale"))
        if scale is None:
            scale = 1.0 + self.weight
            self._scale = scale
        return mx.fast.rms_norm(x, weight=scale, eps=self.eps)


def _norm(args: upstream.TextArgs, dims: int) -> nn.Module:
    if args.use_gemma_norm:
        return GemmaRMSNorm(dims, eps=args.rms_norm_eps)
    return nn.RMSNorm(dims, eps=args.rms_norm_eps)


@partial(mx.compile, shapeless=True)
def _sigmoid_scores(gates: mx.array, bias: mx.array) -> tuple[mx.array, mx.array]:
    scores = mx.sigmoid(gates.astype(mx.float32))
    return scores, -(scores + bias)


@cache
def _normalizer(dtype: mx.Dtype) -> Callable[[mx.array, mx.array, mx.array], mx.array]:
    @partial(mx.compile, shapeless=True)
    def normalize(weights: mx.array, total: mx.array, scaling: mx.array) -> mx.array:
        return (weights / (total + 1e-20) * scaling).astype(dtype)

    return normalize


_ROUTE_SOURCE = """
    // One simdgroup per token. mlx-lm's routing, op for op: sigmoid in float32 (MLX's
    // formula), the bias only for the choice, the TOPK lowest -(score + bias) in a
    // stable ascending sort's order (mlx's argpartition sorts), the chosen scores summed
    // in that order, normalized, scaled and cast.
    constexpr int PER_LANE = E / 32;
    const int row = threadgroup_position_in_grid.y;
    const int lane = thread_index_in_simdgroup;
    threadgroup float score_s[E];
    float choice[PER_LANE];
    for (int i = 0; i < PER_LANE; i++) {
        const int e = lane + 32 * i;
        const float x = gates[(size_t)row * E + e];
        const float y = 1 / (1 + metal::precise::exp(metal::abs(x)));
        const float score = (x < 0) ? y : 1 - y;
        score_s[e] = score;
        choice[i] = -(score + bias[e]);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    int chosen[TOPK];
    for (int k = 0; k < TOPK; k++) {
        float best = INFINITY;
        int best_e = 0x7fffffff;
        for (int i = 0; i < PER_LANE; i++) {
            const int e = lane + 32 * i;
            bool taken = false;
            for (int j = 0; j < k; j++) taken = taken || (chosen[j] == e);
            if (!taken && (choice[i] < best || (choice[i] == best && e < best_e))) {
                best = choice[i];
                best_e = e;
            }
        }
        for (ushort o = 16; o > 0; o /= 2) {
            const float other = simd_shuffle_xor(best, o);
            const int other_e = simd_shuffle_xor(best_e, o);
            if (other < best || (other == best && other_e < best_e)) {
                best = other;
                best_e = other_e;
            }
        }
        chosen[k] = best_e;
    }
    if (lane == 0) {
        float total = 0.0f;
        for (int k = 0; k < TOPK; k++) total += score_s[chosen[k]];
        const float denominator = total + 1e-20f;
        for (int k = 0; k < TOPK; k++) {
            inds[(size_t)row * TOPK + k] = static_cast<uint32_t>(chosen[k]);
            weights[(size_t)row * TOPK + k] =
                static_cast<OutT>(score_s[chosen[k]] / denominator * scaling[0]);
        }
    }
"""


def _route_ops(
    gates: mx.array, bias: mx.array, top_k: int, scaling: float, dtype: mx.Dtype
) -> tuple[mx.array, mx.array]:
    """mlx-lm's routing, op for op: sigmoid in float32, the bias only for the choice,
    the chosen scores normalized, scaled and cast to the activations' dtype. The
    elementwise parts run compiled."""
    scores, choice = _sigmoid_scores(gates, bias)
    chosen = mx.argpartition(choice, kth=top_k - 1, axis=-1)[..., :top_k]
    weights = mx.take_along_axis(scores, chosen, axis=-1)
    total = mx.sum(weights, axis=-1, keepdims=True)
    return chosen, _normalizer(dtype)(weights, total, mx.array(scaling, mx.float32))


def _route(
    gates: mx.array, bias: mx.array, top_k: int, scaling: float, dtype: mx.Dtype
) -> tuple[mx.array, mx.array]:
    """The experts each token goes to and their weights: one kernel per token row when
    it applies (bitwise mlx-lm's ops), mlx-lm's ops otherwise."""
    experts = gates.shape[-1]
    if (
        _route_kernel is None
        or gates.dtype != mx.float32
        or bias.dtype != mx.float32
        or gates.ndim != 2
        or experts % 32 != 0
        or experts > 1024
        or top_k > 32
    ):
        return _route_ops(gates, bias, top_k, scaling, dtype)
    rows = gates.shape[0]
    inds, weights = _route_kernel(
        inputs=[gates, bias, mx.array([scaling], dtype=mx.float32)],
        template=[("E", experts), ("TOPK", top_k), ("OutT", dtype)],
        grid=(32, rows, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(rows, top_k), (rows, top_k)],
        output_dtypes=[mx.uint32, dtype],
    )
    return inds, weights


@partial(mx.compile, shapeless=True)
def _combine(shared: mx.array, *parts: mx.array) -> mx.array:
    """The experts' outputs weighted and summed, plus the shared expert's: mlx-lm's
    ``(y * scores[..., None]).sum(axis=-2) + shared`` in one kernel. Each product is
    rounded to the dtype, and the sum adds them in order, rounding after each add, as
    MLX's reduction over the (small) expert axis does."""
    total = parts[0] * parts[1]
    for i in range(2, len(parts), 2):
        total = total + parts[i] * parts[i + 1]
    return total + shared


class SparseMoeBlock(upstream.SparseMoeBlock):
    def __init__(self, args: upstream.TextArgs):
        super().__init__(args)
        self.top_k = args.num_experts_per_tok

    @override
    def __call__(self, x: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = nn.layers.distributed.sum_gradients(self.sharding_group)(x)

        ne = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])

        shared_out = self.shared_experts(x_flat)
        inds, scores = _route(
            self.gate(x_flat),
            self.e_score_correction_bias,
            self.top_k,
            self.routed_scaling_factor,
            x_flat.dtype,
        )
        inds = mx.stop_gradient(inds)

        y = self.switch_mlp(x_flat, inds)
        if y.dtype in (mx.bfloat16, mx.float16):
            parts: list[mx.array] = []
            for k in range(self.top_k):
                parts += [y[:, k], scores[:, k : k + 1]]
            y = _combine(shared_out, *parts)
        else:
            # In float32 the compiler may fuse a product into the add (no rounding in
            # between), so float32 models keep mlx-lm's ops.
            y = (y * scores[..., None]).sum(axis=-2) + shared_out

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)

        return y.reshape(*ne, -1)


_BLOCK_SCORES_SOURCE = """
    // One threadgroup per (sequence, key block), one thread per key: the key's float32
    // score against every index head (its bf16 key times the float32 query, summed over
    // the head dim in order), then each head's best score in the block. The query's own
    // blocks score +inf (always kept), as in mlx-lm.
    const int S = params[0];
    const int capacity = params[1];
    const int n_blocks = params[2];
    const int blk = threadgroup_position_in_grid.y;
    const int b = threadgroup_position_in_grid.z;
    const int t = thread_position_in_threadgroup.x;
    const int lane = thread_index_in_simdgroup;
    const int sg = simdgroup_index_in_threadgroup;
    constexpr int SIMDS = BLOCK / 32;

    threadgroup float q_s[H * D];
    for (int i = t; i < H * D; i += BLOCK) q_s[i] = iq[(size_t)b * H * D + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int pos = blk * BLOCK + t;
    bool ok = pos < S;
    if (HAS_MASK && ok) ok = mask[(size_t)b * S + pos];
    float acc[H];
    for (int h = 0; h < H; h++) acc[h] = 0.0f;
    if (ok) {
        const device T* kp = ikeys + ((size_t)b * capacity + pos) * D;
        for (int d = 0; d < D; d++) {
            const float kd = static_cast<float>(kp[d]);
            for (int h = 0; h < H; h++) acc[h] += q_s[h * D + d] * kd;
        }
    }
    threadgroup float red[H * SIMDS];
    for (int h = 0; h < H; h++) {
        float m = ok ? acc[h] : -INFINITY;
        m = simd_max(m);
        if (lane == 0) red[h * SIMDS + sg] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < H) {
        float m = -INFINITY;
        for (int i = 0; i < SIMDS; i++) m = max(m, red[t * SIMDS + i]);
        const int q_block = (S - 1) / BLOCK;
        const bool local = blk <= q_block && blk > q_block - LOCAL;
        block_max[((size_t)b * H + t) * n_blocks + blk] = local ? INFINITY : m;
    }
"""

_ATTEND_TO_BLOCKS_SOURCE = """
    // One threadgroup per (sequence, KV group, selected block): its HQ / H simdgroups
    // are the group's query heads, so the block's keys and values are read from memory
    // once and shared through the core's cache. 8 lanes per key (D / 8 dims each, one
    // contiguous row per 8 lanes), 4 keys at a time, with a fixed trip count so that
    // the loads pipeline. Writes the block's partial softmax (max, sum, weighted
    // values) for each query head.
    const int S = params[0];
    const int capacity = params[1];
    const float scale = scale_arr[0];
    const int row = threadgroup_position_in_grid.y;  // (b * H + h) * TOPK + j
    const int bh_kv = row / TOPK;
    const int j_blk = row % TOPK;
    const int b = bh_kv / H;
    const int h = bh_kv % H;
    const int lane = thread_index_in_simdgroup;
    const int sg = simdgroup_index_in_threadgroup;
    const int hq = h * (HQ / H) + sg;
    constexpr int DPL = D / 8;
    const int group = lane / 8;
    const int sub = lane % 8;

    float qv[DPL];
    const size_t q_off = (size_t)(b * HQ + hq) * D + sub * DPL;
    for (int i = 0; i < DPL; i++) qv[i] = scale * static_cast<float>(q[q_off + i]);
    // The TOPK blocks this KV group keeps: a block's rank in a stable ascending sort
    // of the negated scores (mlx's argpartition sorts): higher scores first, ties to the
    // lower block. Every threadgroup of the group computes the same choice.
    const int n_blocks = params[2];
    const int tid = thread_position_in_threadgroup.x;
    const device float* bm = block_max + (size_t)bh_kv * n_blocks;
    threadgroup int chosen[TOPK];
    for (int i = tid; i < n_blocks; i += 32 * (HQ / H)) {
        const float x = bm[i];
        int rank = 0;
        for (int j = 0; j < n_blocks; j++) {
            const float y = bm[j];
            rank += (y > x || (y == x && j < i)) ? 1 : 0;
        }
        if (rank < TOPK) chosen[rank] = i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        // in position order
        for (int i = 1; i < TOPK; i++) {
            const int x = chosen[i];
            int j = i - 1;
            while (j >= 0 && chosen[j] > x) { chosen[j + 1] = chosen[j]; j--; }
            chosen[j + 1] = x;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int blk = chosen[j_blk];
    const device T* kbase = keys + ((size_t)bh_kv * capacity) * D + sub * DPL;
    const device T* vbase = values + ((size_t)bh_kv * capacity) * D + sub * DPL;
    const int first = blk * BLOCK;

    threadgroup float p_s[(HQ / H) * BLOCK];
    #pragma unroll(4)
    for (int j0 = 0; j0 < BLOCK; j0 += 4) {
        const int j = j0 + group;
        const int pos = first + j;
        bool ok = pos < S;
        if (HAS_MASK && ok) ok = mask[(size_t)b * S + pos];
        const device T* kp = kbase + (size_t)min(pos, capacity - 1) * D;
        float acc = 0.0f;
        #pragma unroll
        for (int i = 0; i < DPL; i++) acc += qv[i] * static_cast<float>(kp[i]);
        acc += simd_shuffle_xor(acc, (ushort)1);
        acc += simd_shuffle_xor(acc, (ushort)2);
        acc += simd_shuffle_xor(acc, (ushort)4);
        if (sub == 0) p_s[sg * BLOCK + j] = ok ? acc : -INFINITY;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    float m = -INFINITY;
    for (int j = lane; j < BLOCK; j += 32) m = max(m, p_s[sg * BLOCK + j]);
    m = simd_max(m);
    float l = 0.0f;
    for (int j = lane; j < BLOCK; j += 32) {
        const float x = p_s[sg * BLOCK + j];
        const float p = (x == -INFINITY) ? 0.0f : fast::exp(x - m);
        p_s[sg * BLOCK + j] = p;
        l += p;
    }
    l = simd_sum(l);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    float o[DPL];
    for (int i = 0; i < DPL; i++) o[i] = 0.0f;
    #pragma unroll(4)
    for (int j0 = 0; j0 < BLOCK; j0 += 4) {
        const int j = j0 + group;
        const float p = p_s[sg * BLOCK + j];
        const device T* vp = vbase + (size_t)min(first + j, capacity - 1) * D;
        #pragma unroll
        for (int i = 0; i < DPL; i++) o[i] += p * static_cast<float>(vp[i]);
    }
    for (int i = 0; i < DPL; i++) {
        o[i] += simd_shuffle_xor(o[i], (ushort)8);
        o[i] += simd_shuffle_xor(o[i], (ushort)16);
    }

    const size_t part = (size_t)(b * HQ + hq) * TOPK + j_blk;
    if (lane == 0) { part_m[part] = m; part_l[part] = l; }
    if (group == 0) {
        for (int i = 0; i < DPL; i++) part_o[part * D + sub * DPL + i] = o[i];
    }
"""

_MERGE_BLOCKS_SOURCE = """
    // One threadgroup per (sequence, query head), one thread per output dim: the
    // TOPK blocks' partial softmaxes merged.
    const int bh = threadgroup_position_in_grid.y;
    const int d = thread_position_in_threadgroup.x;
    float m_all = -INFINITY;
    for (int j = 0; j < TOPK; j++) m_all = max(m_all, part_m[bh * TOPK + j]);
    float num = 0.0f;
    float den = 0.0f;
    for (int j = 0; j < TOPK; j++) {
        const float mj = part_m[bh * TOPK + j];
        if (mj == -INFINITY) continue;
        const float w = fast::exp(mj - m_all);
        num += w * part_o[((size_t)bh * TOPK + j) * D + d];
        den += w * part_l[bh * TOPK + j];
    }
    out[(size_t)bh * D + d] = static_cast<T>(den > 0.0f ? num / den : 0.0f);
"""


def _make_kernel(name: str, inputs: list[str], outputs: list[str], source: str) -> Any:
    if not mx.metal.is_available():
        return None
    return mx.fast.metal_kernel(
        name=name, input_names=inputs, output_names=outputs, source=source
    )


_route_kernel = _make_kernel(
    "minimax_m3_route",
    ["gates", "bias", "scaling"],
    ["inds", "weights"],
    _ROUTE_SOURCE,
)
_block_scores_kernel = _make_kernel(
    "minimax_m3_block_scores",
    ["iq", "ikeys", "mask", "params"],
    ["block_max"],
    _BLOCK_SCORES_SOURCE,
)
# The selection reads every block's score in each threadgroup; longer contexts (over
# 512k keys with 128-key blocks) take the MLX-ops path.
_MAX_BLOCKS = 4096
_attend_to_blocks_kernel = _make_kernel(
    "minimax_m3_attend_to_blocks",
    ["q", "keys", "values", "block_max", "mask", "params", "scale_arr"],
    ["part_m", "part_l", "part_o"],
    _ATTEND_TO_BLOCKS_SOURCE,
)
_merge_blocks_kernel = _make_kernel(
    "minimax_m3_merge_blocks",
    ["part_m", "part_l", "part_o"],
    ["out"],
    _MERGE_BLOCKS_SOURCE,
)


def _sparse_decode_kernels_apply(
    q: mx.array,
    keys: mx.array,
    values: mx.array,
    index_keys: mx.array,
    topk: int,
    block_size: int,
    n_keys: int,
) -> bool:
    return (
        _block_scores_kernel is not None
        and _attend_to_blocks_kernel is not None
        and _merge_blocks_kernel is not None
        and q.dtype in (mx.bfloat16, mx.float16, mx.float32)
        and keys.dtype == q.dtype
        and values.dtype == q.dtype
        and q.shape[-1] % 32 == 0
        and q.shape[-1] == keys.shape[-1] == values.shape[-1]
        and q.shape[-1] <= 1024
        and q.shape[1] % keys.shape[1] == 0
        and q.shape[1] // keys.shape[1] <= 32
        and index_keys.shape[1] == 1
        and 1 <= topk <= 32
        and block_size % 32 == 0
        and block_size <= 1024
        and -(-n_keys // block_size) <= _MAX_BLOCKS
    )


def _sparse_decode_kernels(
    q: mx.array,
    index_q: mx.array,
    keys: mx.array,
    values: mx.array,
    index_keys: mx.array,
    mask: mx.array | None,
    n_keys: int,
    block_size: int,
    topk: int,
    local_blocks: int,
    scale: float,
) -> mx.array:
    """A one-token step over the selected key blocks: three kernels. The cache arrays
    are the whole (B, heads, capacity, D) buffers, read in place up to S keys."""
    batch, q_heads, _, head_dim = q.shape
    heads = keys.shape[1]
    n_blocks = -(-n_keys // block_size)
    if mask is None:
        mask_arr = mx.zeros((1,), dtype=mx.bool_)
    else:
        mask_arr = mx.broadcast_to(mask, (batch, 1, 1, n_keys))
    (block_max,) = _block_scores_kernel(
        inputs=[
            index_q,
            index_keys,
            mask_arr,
            mx.array([n_keys, index_keys.shape[2], n_blocks], dtype=mx.int32),
        ],
        template=[
            ("T", index_keys.dtype),
            ("D", index_q.shape[-1]),
            ("H", index_q.shape[1]),
            ("BLOCK", block_size),
            ("LOCAL", local_blocks),
            ("HAS_MASK", mask is not None),
        ],
        grid=(block_size, n_blocks, batch),
        threadgroup=(block_size, 1, 1),
        output_shapes=[(batch, index_q.shape[1], n_blocks)],
        output_dtypes=[mx.float32],
    )
    part_m, part_l, part_o = _attend_to_blocks_kernel(
        inputs=[
            q,
            keys,
            values,
            block_max,
            mask_arr,
            mx.array([n_keys, keys.shape[2], n_blocks], dtype=mx.int32),
            mx.array([scale], dtype=mx.float32),
        ],
        template=[
            ("T", q.dtype),
            ("D", head_dim),
            ("H", heads),
            ("HQ", q_heads),
            ("BLOCK", block_size),
            ("TOPK", topk),
            ("HAS_MASK", mask is not None),
        ],
        grid=(32 * (q_heads // heads), batch * heads * topk, 1),
        threadgroup=(32 * (q_heads // heads), 1, 1),
        output_shapes=[
            (batch, q_heads, topk),
            (batch, q_heads, topk),
            (batch, q_heads, topk, head_dim),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    (out,) = _merge_blocks_kernel(
        inputs=[part_m, part_l, part_o],
        template=[("T", q.dtype), ("D", head_dim), ("TOPK", topk)],
        grid=(head_dim, batch * q_heads, 1),
        threadgroup=(head_dim, 1, 1),
        output_shapes=[(batch, q_heads, 1, head_dim)],
        output_dtypes=[q.dtype],
    )
    return out


class Attention(upstream.Attention):
    def __init__(self, args: upstream.TextArgs, layer_idx: int):
        super().__init__(args, layer_idx)
        if self.use_qk_norm:
            self.q_norm = _norm(args, self.head_dim)
            self.k_norm = _norm(args, self.head_dim)
        if self.is_sparse_attn:
            self.index_q_norm = _norm(args, self.index_head_dim)
            self.index_k_norm = _norm(args, self.index_head_dim)

    def _index_keys(self, x: mx.array, offset: Any, cache: Any) -> mx.array:
        """This step's index keys, appended to the index-key cache."""
        batch, length, _ = x.shape
        k = self.index_k_proj(x).reshape(batch, length, 1, self.index_head_dim)
        k = self.index_k_norm(k).transpose(0, 2, 1, 3)
        k = self.index_rope(k, offset=offset)
        if cache is not None:
            k, _ = cache.update_and_fetch(k, mx.zeros((batch, 1, length, 0), k.dtype))
        return k

    def _index_queries(self, x: mx.array, offset: Any) -> mx.array:
        batch, length, _ = x.shape
        heads, head_dim = self.index_n_heads, self.index_head_dim
        q = self.index_q_proj(x).reshape(batch, length, heads, head_dim)
        q = self.index_q_norm(q).transpose(0, 2, 1, 3)
        return self.index_rope(q, offset=offset).astype(mx.float32)

    def _index_scores(
        self, index_q: mx.array, k: mx.array, per_key: mx.array | None
    ) -> mx.array:
        """Each query's float32 score for each cached key: mlx-lm's, op for op."""
        scores = index_q @ k.astype(mx.float32).swapaxes(-1, -2)
        if per_key is not None:
            # A future (or padding) key must not win a block, so mask before pooling.
            scores = mx.where(per_key, scores, -mx.inf)
        return scores

    def _block_scores(self, scores: mx.array) -> mx.array:
        """Each query's score for each key block (its best key's): mlx-lm's, op for op."""
        batch, heads, length, n_keys = scores.shape
        block_size = self.index_block_size
        full_blocks, tail_keys = divmod(n_keys, block_size)
        n_blocks = full_blocks + bool(tail_keys)
        split = full_blocks * block_size
        block_scores = scores[..., :split].reshape(
            batch, heads, length, full_blocks, block_size
        )
        block_scores = block_scores.max(axis=-1)
        if tail_keys:
            tail_score = scores[..., split:].max(axis=-1, keepdims=True)
            block_scores = mx.concatenate([block_scores, tail_score], axis=-1)
        if self.index_local_blocks > 0:
            qpos = mx.arange(n_keys - length, n_keys)
            q_block = (qpos // block_size)[:, None]
            ids = mx.arange(n_blocks)
            is_local = (ids <= q_block) & (ids > q_block - self.index_local_blocks)
            block_scores = mx.where(is_local, mx.inf, block_scores)
        return block_scores

    def _select(self, block_scores: mx.array) -> mx.array:
        topk = self.index_topk_blocks
        inds = mx.argpartition(-block_scores, kth=topk - 1, axis=-1)[..., :topk]
        return mx.stop_gradient(inds)

    def _sparse_decode_ops(
        self,
        q: mx.array,
        keys: mx.array,
        values: mx.array,
        block_scores: mx.array,
        mask: mx.array | None,
    ) -> mx.array:
        """The sparse decode step in MLX ops: gather the selected blocks, then attend."""
        batch, q_heads, _, _ = q.shape
        heads = keys.shape[1]
        n_keys = keys.shape[2]
        block_size = self.index_block_size
        topk = self.index_topk_blocks
        # Blocks in position order; each block's keys in order.
        blocks = mx.sort(self._select(block_scores), axis=-1)  # (B, H, 1, topk)
        positions = blocks[..., None] * block_size + mx.arange(block_size)
        positions = positions.reshape(batch, heads, topk * block_size)
        valid = positions < n_keys
        positions = mx.minimum(positions, n_keys - 1)
        k = mx.take_along_axis(keys, positions[..., None], axis=2)
        v = mx.take_along_axis(values, positions[..., None], axis=2)
        keep = valid[:, :, None, :]
        if mask is not None:
            keep = keep & mx.take_along_axis(
                mx.broadcast_to(mask, (batch, heads, 1, n_keys)),
                positions[:, :, None, :],
                axis=-1,
            )
        keep = mx.repeat(keep, q_heads // heads, axis=1)
        return mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=keep
        )

    @override
    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        batch, length, _ = x.shape
        kv_cache = cache[0] if cache is not None else None
        if hasattr(kv_cache, "bits"):
            # A quantized KV cache: mlx-lm's path (exo doesn't make one for this model).
            return super().__call__(x, mask, cache)
        offset = kv_cache.offset if kv_cache is not None else 0

        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = q.reshape(batch, length, self.num_attention_heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.reshape(batch, length, self.num_key_value_heads, self.head_dim)
        k = k.transpose(0, 2, 1, 3)
        v = v.reshape(batch, length, self.num_key_value_heads, self.head_dim)
        v = v.transpose(0, 2, 1, 3)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        # Select before the cache advances: it moves its offset array in place.
        sparse = False
        per_key = None
        index_cache = None
        index_keys = None
        index_q = None
        n_keys = 0
        if self.is_sparse_attn:
            index_cache = cache[1] if cache is not None else None
            index_keys = self._index_keys(x, offset, index_cache)
            n_keys = index_keys.shape[2]
            n_blocks = -(-n_keys // self.index_block_size)
            if self.index_topk_blocks < n_blocks:
                sparse = True
                if isinstance(mask, mx.array):
                    per_key = mask
                elif length > 1:
                    # Blocks are cut over cache slots, so place queries by slot.
                    per_key = (
                        mx.arange(n_keys)[None]
                        <= mx.arange(n_keys - length, n_keys)[:, None]
                    )
                # A single query sees every cached key: the causal mask is all true.
                index_q = self._index_queries(x, offset)

        if kv_cache is not None:
            k, v = kv_cache.update_and_fetch(k, v)
            # Keep the indexer cache in the graph: below the block budget nothing
            # consumes its keys, and nothing ever consumes its values.
            if self.is_sparse_attn:
                index_kv = [cache[1].keys, cache[1].values]
                kv_cache.keys = mx.depends(kv_cache.keys, index_kv)

        if not sparse or index_q is None or index_keys is None:
            out = scaled_dot_product_attention(
                q, k, v, cache=kv_cache, scale=self.scale, mask=mask
            )
        elif (
            length == 1
            and kv_cache is not None
            and index_cache is not None
            and _sparse_decode_kernels_apply(
                q,
                kv_cache.keys,
                kv_cache.values,
                index_cache.keys,
                self.index_topk_blocks,
                self.index_block_size,
                n_keys,
            )
        ):
            out = _sparse_decode_kernels(
                q,
                index_q,
                kv_cache.keys,
                kv_cache.values,
                index_cache.keys,
                per_key,
                n_keys,
                self.index_block_size,
                self.index_topk_blocks,
                self.index_local_blocks,
                self.scale,
            )
        else:
            block_scores = self._block_scores(
                self._index_scores(index_q, index_keys, per_key)
            )
            if length == 1:
                out = self._sparse_decode_ops(q, k, v, block_scores, per_key)
            else:
                out = self._sparse_prefill(q, k, v, block_scores, per_key)
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(out)

    def _sparse_prefill(
        self,
        q: mx.array,
        keys: mx.array,
        values: mx.array,
        block_scores: mx.array,
        per_key: mx.array | None,
    ) -> mx.array:
        """mlx-lm's selection mask, applied per KV group: the query heads of each group
        and its keys form one batch entry, so the mask needs no copy per query head."""
        batch, q_heads, length, head_dim = q.shape
        heads = keys.shape[1]
        n_keys = keys.shape[2]
        block_size = self.index_block_size
        inds = self._select(block_scores)
        keep = mx.put_along_axis(
            mx.zeros(block_scores.shape, dtype=mx.bool_),
            inds,
            mx.array(True),
            axis=-1,
        )
        # A kept block still holds future keys, so mask again.
        keep = mx.repeat(keep, block_size, axis=-1)[..., :n_keys]
        if per_key is not None:
            keep = keep & per_key
        out = mx.fast.scaled_dot_product_attention(
            q.reshape(batch * heads, q_heads // heads, length, head_dim),
            keys.reshape(batch * heads, 1, n_keys, head_dim),
            values.reshape(batch * heads, 1, n_keys, values.shape[-1]),
            scale=self.scale,
            mask=keep.reshape(batch * heads, 1, length, n_keys),
        )
        return out.reshape(batch, q_heads, length, -1)


class DecoderLayer(upstream.DecoderLayer):
    def __init__(self, args: upstream.TextArgs, layer_idx: int):
        # mlx-lm's constructor with this module's classes (not called, so nothing is
        # built twice).
        nn.Module.__init__(self)
        self.self_attn = Attention(args, layer_idx)
        self.is_moe = args.is_moe(layer_idx)
        if self.is_moe:
            self.block_sparse_moe = SparseMoeBlock(args)
        else:
            self.mlp = upstream.DenseMLP(
                args.hidden_size,
                args.dense_intermediate_size,
                activation=SwigluOAI(alpha=args.swiglu_alpha, limit=args.swiglu_limit),
            )
        self.input_layernorm = _norm(args, args.hidden_size)
        self.post_attention_layernorm = _norm(args, args.hidden_size)


class TextModel(upstream.TextModel):
    def __init__(self, args: upstream.TextArgs):
        # PipelineMixin's and nn.Module's constructors, then mlx-lm's fields.
        super(upstream.TextModel, self).__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = _norm(args, args.hidden_size)


class LanguageModel(upstream.LanguageModel):
    def __init__(self, config: upstream.TextArgs):
        nn.Module.__init__(self)
        self.args = config
        self.model = TextModel(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)


class Model(upstream.Model):
    def __init__(self, config: upstream.ModelArgs):
        nn.Module.__init__(self)
        self.args = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(cast(upstream.TextArgs, config.text_config))
