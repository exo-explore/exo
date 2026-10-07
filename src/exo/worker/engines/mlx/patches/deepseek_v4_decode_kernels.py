"""Fused Metal kernels for the DeepSeek V4 single-token decode step.

At decode every DeepSeek V4 layer is a long chain of tiny GPU kernels, and on
Apple GPUs each dependent dispatch costs a few microseconds regardless of how
little work it does. These kernels collapse the chains that dominate a layer:

* ``hc_pre``: cast + RMSNorm + mix matmul become one kernel (the RMS factor is
  applied to the dot products instead of the inputs).
* ``hc_post``: cast + comb matmul + post blend become one kernel.
* Attention q / kv: RMSNorm + RoPE on the trailing dims + concatenate become
  one kernel each, and the inverse RoPE on the attention output is one kernel.
* Attention itself: the sink-aware SDPA, which MLX runs as matmul + concat +
  softmax + matmul for this head size, becomes one kernel.

Only the uniform single-token decode path is replaced; prefill, per-row
offsets and unsupported shapes fall back to the original mlx_lm code.
"""

from typing import cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import deepseek_v4
from mlx_lm.models.deepseek_v4 import (
    _K_COMP,  # pyright: ignore[reportPrivateUsage]
    DeepseekV4Cache,
    HyperConnection,
    V4Attention,
)

_THREADGROUP = 256
_ATTENTION_THREADS = 256

_HC_MIX_SOURCE = """
    uint group = threadgroup_position_in_grid.x;
    uint row = group / MIX;
    uint mix = group % MIX;
    uint t = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint simdgroup = simdgroup_index_in_threadgroup;

    threadgroup float partial_dots[8];
    threadgroup float partial_squares[8];

    float dot = 0.0f;
    float square = 0.0f;
    for (uint k = t; k < K; k += 256) {
        float v = static_cast<float>(x[row * K + k]);
        dot += v * fn[mix * K + k];
        square += v * v;
    }
    dot = simd_sum(dot);
    square = simd_sum(square);
    if (lane == 0) {
        partial_dots[simdgroup] = dot;
        partial_squares[simdgroup] = square;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        float total_dot = 0.0f;
        float total_square = 0.0f;
        for (int i = 0; i < 8; ++i) {
            total_dot += partial_dots[i];
            total_square += partial_squares[i];
        }
        mixes[row * MIX + mix] = total_dot * metal::rsqrt(total_square / float(K) + eps[0]);
    }
"""

_HC_POST_SOURCE = """
    uint gid = thread_position_in_grid.x;
    uint n = gid / D;
    uint d = gid % D;
    float f = static_cast<float>(f_out[n * D + d]);
    float r[HC];
    for (int i = 0; i < HC; ++i) {
        r[i] = static_cast<float>(residual[(n * HC + i) * D + d]);
    }
    for (int j = 0; j < HC; ++j) {
        float y = post[n * HC + j] * f;
        for (int i = 0; i < HC; ++i) {
            y += comb[(n * HC + i) * HC + j] * r[i];
        }
        out[(n * HC + j) * D + d] = static_cast<OUT_T>(y);
    }
"""

# One threadgroup per row of D elements, one thread per (even, odd) pair.
_NORM_ROPE_SOURCE = """
    uint row = threadgroup_position_in_grid.x;
    uint pair = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint simdgroup = simdgroup_index_in_threadgroup;
    constexpr uint NUM_SIMDGROUPS = (D / 2) / 32;
    threadgroup float partial[NUM_SIMDGROUPS];

    float a = static_cast<float>(x[row * D + 2 * pair]);
    float b = static_cast<float>(x[row * D + 2 * pair + 1]);

    if (NORM) {
        float square = simd_sum(a * a + b * b);
        if (lane == 0) {
            partial[simdgroup] = square;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = 0.0f;
        for (uint i = 0; i < NUM_SIMDGROUPS; ++i) {
            total += partial[i];
        }
        float inverse_rms = metal::rsqrt(total / float(D) + eps[0]);
        a = static_cast<float>(static_cast<OUT_T>(a * inverse_rms));
        b = static_cast<float>(static_cast<OUT_T>(b * inverse_rms));
        if (HAS_WEIGHT) {
            a = static_cast<float>(static_cast<OUT_T>(a * static_cast<float>(weight[2 * pair])));
            b = static_cast<float>(static_cast<OUT_T>(b * static_cast<float>(weight[2 * pair + 1])));
        }
    }

    constexpr uint FIRST_ROTATED_PAIR = (D - ROPE_DIMS) / 2;
    if (pair >= FIRST_ROTATED_PAIR) {
        float theta = position[0] / freqs[pair - FIRST_ROTATED_PAIR];
        float c = metal::cos(theta);
        float s = metal::sin(theta);
        float rotated_a = a * c - b * s;
        float rotated_b = a * s + b * c;
        a = rotated_a;
        b = rotated_b;
    }
    out[row * D + 2 * pair] = static_cast<OUT_T>(a);
    out[row * D + 2 * pair + 1] = static_cast<OUT_T>(b);
"""

# One threadgroup per (batch, head). Keys and values are the same tensor, so
# each simdgroup streams its share of the keys once, keeping a running
# (online) softmax and the weighted sum of the same registers it scored with.
# The simdgroups' partial results are merged at the end.
_ATTENTION_SOURCE = """
    uint batch_head = threadgroup_position_in_grid.x;
    uint batch = batch_head / H;
    uint head = batch_head % H;
    uint lane = thread_index_in_simdgroup;
    uint simdgroup = simdgroup_index_in_threadgroup;
    constexpr uint PER_LANE = D / 32;
    constexpr uint SIMDGROUPS = THREADS / 32;
    int keys = kv_shape[1];

    threadgroup float partial_max[SIMDGROUPS];
    threadgroup float partial_sum[SIMDGROUPS];
    threadgroup float partial_out[SIMDGROUPS * D];

    float query[PER_LANE];
    for (uint i = 0; i < PER_LANE; ++i) {
        query[i] = static_cast<float>(q[batch_head * D + lane * PER_LANE + i]) * scale[0];
    }
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float acc[PER_LANE];
    for (uint i = 0; i < PER_LANE; ++i) {
        acc[i] = 0.0f;
    }
    for (int k = simdgroup; k < keys; k += SIMDGROUPS) {
        float value[PER_LANE];
        float dot = 0.0f;
        for (uint i = 0; i < PER_LANE; ++i) {
            value[i] = static_cast<float>(kv[(batch * keys + k) * D + lane * PER_LANE + i]);
            dot += query[i] * value[i];
        }
        float score = simd_sum(dot);
        float new_max = metal::max(running_max, score);
        float correction = metal::exp(running_max - new_max);
        float weight = metal::exp(score - new_max);
        running_sum = running_sum * correction + weight;
        for (uint i = 0; i < PER_LANE; ++i) {
            acc[i] = acc[i] * correction + weight * value[i];
        }
        running_max = new_max;
    }
    if (lane == 0) {
        partial_max[simdgroup] = running_max;
        partial_sum[simdgroup] = running_sum;
    }
    for (uint i = 0; i < PER_LANE; ++i) {
        partial_out[simdgroup * D + lane * PER_LANE + i] = acc[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sink = static_cast<float>(sinks[head]);
    float total_max = sink;
    for (uint g = 0; g < SIMDGROUPS; ++g) {
        total_max = metal::max(total_max, partial_max[g]);
    }
    float denominator = metal::exp(sink - total_max);
    for (uint g = 0; g < SIMDGROUPS; ++g) {
        denominator += partial_sum[g] * metal::exp(partial_max[g] - total_max);
    }
    uint t = thread_position_in_threadgroup.x;
    for (uint d = t; d < D; d += THREADS) {
        float total = 0.0f;
        for (uint g = 0; g < SIMDGROUPS; ++g) {
            total += partial_out[g * D + d] * metal::exp(partial_max[g] - total_max);
        }
        out[batch_head * D + d] = static_cast<OUT_T>(total / denominator);
    }
"""

_hc_mix_kernel = mx.fast.metal_kernel(
    name="dsv4_hc_mix",
    input_names=["x", "fn", "eps"],
    output_names=["mixes"],
    source=_HC_MIX_SOURCE,
)
_hc_post_kernel = mx.fast.metal_kernel(
    name="dsv4_hc_post",
    input_names=["f_out", "residual", "post", "comb"],
    output_names=["out"],
    source=_HC_POST_SOURCE,
)
_norm_rope_kernel = mx.fast.metal_kernel(
    name="dsv4_norm_rope",
    input_names=["x", "weight", "eps", "position", "freqs"],
    output_names=["out"],
    source=_NORM_ROPE_SOURCE,
)
_attention_kernel = mx.fast.metal_kernel(
    name="dsv4_decode_attention",
    input_names=["q", "kv", "sinks", "scale"],
    output_names=["out"],
    source=_ATTENTION_SOURCE,
)

_scalars: dict[float, mx.array] = {}


def _scalar(value: float) -> mx.array:
    """An evaluated one-element float32 array, cached for constants like eps."""
    array = _scalars.get(value)
    if array is None:
        array = mx.array([value], dtype=mx.float32)
        mx.eval(array)
        _scalars[value] = array
    return array


def hc_mix(x: mx.array, fn: mx.array, eps: float) -> mx.array:
    """``rms_norm(x.astype(float32)) @ fn.T`` for x ``[..., K]`` and fn ``[MIX, K]``."""
    rows = x.size // x.shape[-1]
    mix, k = fn.shape
    return _hc_mix_kernel(
        inputs=[x, fn, _scalar(eps)],
        template=[("MIX", mix), ("K", k)],
        grid=(rows * mix * _THREADGROUP, 1, 1),
        threadgroup=(_THREADGROUP, 1, 1),
        output_shapes=[(*x.shape[:-1], mix)],
        output_dtypes=[mx.float32],
    )[0]


def hc_post(
    f_out: mx.array, residual: mx.array, post: mx.array, comb: mx.array
) -> mx.array:
    """``post * f_out + comb^T @ residual`` (see ``_hc_expand_ops``)."""
    hc, d = residual.shape[-2], residual.shape[-1]
    return _hc_post_kernel(
        inputs=[f_out, residual, post, comb],
        template=[("HC", hc), ("D", d), ("OUT_T", f_out.dtype)],
        grid=(f_out.size, 1, 1),
        threadgroup=(_THREADGROUP, 1, 1),
        output_shapes=[residual.shape],
        output_dtypes=[f_out.dtype],
    )[0]


def norm_rope(
    x: mx.array,
    weight: mx.array | None,
    eps: float | None,
    position: float,
    freqs: mx.array,
    rope_dims: int,
) -> mx.array:
    """Optional RMSNorm (optionally weighted) then RoPE on the last ``rope_dims``
    of each row, matching ``mx.fast.rms_norm`` followed by ``mx.fast.rope``
    with ``traditional=True`` at ``position`` (already multiplied by the scale)."""
    d = x.shape[-1]
    rows = x.size // d
    return _norm_rope_kernel(
        inputs=[
            x,
            weight if weight is not None else _scalar(0.0),
            _scalar(eps if eps is not None else 0.0),
            mx.array([position], dtype=mx.float32),
            freqs,
        ],
        template=[
            ("D", d),
            ("ROPE_DIMS", rope_dims),
            ("NORM", eps is not None),
            ("HAS_WEIGHT", weight is not None),
            ("OUT_T", x.dtype),
        ],
        grid=(rows * (d // 2), 1, 1),
        threadgroup=(d // 2, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


def decode_attention(
    q: mx.array, kv: mx.array, sinks: mx.array, scale: float
) -> mx.array:
    """Single-query attention with sinks. q ``[B, H, 1, D]``, kv ``[B, L, D]``."""
    batch, heads, _, d = q.shape
    return _attention_kernel(
        inputs=[q, kv, sinks, _scalar(scale)],
        template=[
            ("H", heads),
            ("D", d),
            ("THREADS", _ATTENTION_THREADS),
            ("OUT_T", q.dtype),
        ],
        grid=(batch * heads * _ATTENTION_THREADS, 1, 1),
        threadgroup=(_ATTENTION_THREADS, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )[0]


_original_hc_pre = HyperConnection.hc_pre
_original_hc_expand_ops = deepseek_v4._hc_expand_ops  # pyright: ignore[reportPrivateUsage]
_original_attention = V4Attention.__call__


def _patched_hc_pre(
    self: HyperConnection, x: mx.array
) -> tuple[mx.array, mx.array, mx.array]:
    batch, sequence, hc, d = x.shape
    if sequence != 1 or x.dtype != mx.bfloat16 or (hc * d) % _THREADGROUP != 0:
        return _original_hc_pre(self, x)
    mixes = hc_mix(x.reshape(batch, sequence, hc * d), self.fn, self.norm_eps)
    return deepseek_v4.hc_sinkhorn_collapse(
        mixes,
        self.scale,
        self.base,
        x,
        self.hc_mult,
        self.sinkhorn_iters,
        self._eps_arr,
    )


def _patched_hc_expand_ops(
    f_out: mx.array, residual: mx.array, post: mx.array, comb: mx.array
) -> mx.array:
    if f_out.shape[1] != 1 or f_out.dtype != mx.bfloat16:
        return _original_hc_expand_ops(f_out, residual, post, comb)
    return hc_post(f_out, residual, post, comb)


def _decode_supported(attention: V4Attention, x: mx.array, cache: object) -> bool:
    if x.shape[1] != 1 or x.dtype != mx.bfloat16:
        return False
    if not isinstance(cache, DeepseekV4Cache):
        return False
    if isinstance(cache.local.offset, mx.array):
        return False
    wqkv = attention.wqkv_a
    if isinstance(wqkv, nn.QuantizedLinear) and wqkv.mode == "mxfp4":
        return False
    wq_b = attention.wq_b
    if isinstance(wq_b, nn.QuantizedLinear) and wq_b.mode == "mxfp4":
        return False
    head_dim = attention.head_dim
    return head_dim % 64 == 0 and head_dim // 2 <= 1024


def _patched_attention(
    self: V4Attention, x: mx.array, cache: object = None
) -> mx.array:
    if not _decode_supported(self, x, cache):
        return _original_attention(self, x, cache=cache)
    v4_cache = cast(DeepseekV4Cache, cache)
    batch = x.shape[0]
    rope_dims = self.rope_head_dim
    head_dim = self.head_dim
    freqs = self.rope.freqs
    win_cache = v4_cache.local
    offset = win_cache.offset

    qkv_a = self.wqkv_a(x)
    qr = mx.fast.rms_norm(qkv_a[..., : self.q_lora_rank], self.q_norm.weight, self.eps)
    q_flat = self.wq_b(qr)
    q = norm_rope(
        q_flat.reshape(batch * self.n_heads, head_dim),
        None,
        self.eps,
        float(offset),
        freqs,
        rope_dims,
    ).reshape(batch, self.n_heads, 1, head_dim)
    kv = norm_rope(
        qkv_a[..., self.q_lora_rank :].reshape(batch, head_dim),
        self.kv_norm.weight,
        self.eps,
        float(offset),
        freqs,
        rope_dims,
    ).reshape(batch, 1, head_dim)

    k4 = kv[:, None, :, :]
    window_keys, _ = cast(
        tuple[mx.array, mx.array],
        win_cache.update_and_fetch(k4, k4),  # pyright: ignore[reportUnknownMemberType]
    )
    window_kv = window_keys.squeeze(1)
    indexer_topk: mx.array | None = None
    compressed: mx.array | None = None
    if self.compress_ratio:
        self.compressor(x, v4_cache, offset, key=_K_COMP)
        if self.compress_ratio == 4:
            indexer_topk = self.indexer(x, qr, v4_cache, offset)
        compressed = v4_cache.get_branch(_K_COMP).pool

    if compressed is not None and compressed.shape[1] > 0:
        if indexer_topk is not None:
            d = compressed.shape[-1]
            pool_len = compressed.shape[1]
            expanded = mx.broadcast_to(
                compressed[:, None, None, :, :], (batch, 1, 1, pool_len, d)
            )
            index = mx.broadcast_to(
                indexer_topk[:, None, :, :, None],
                (batch, 1, 1, indexer_topk.shape[-1], d),
            )
            gathered = mx.take_along_axis(expanded, index, axis=3).reshape(batch, -1, d)
            kv_all = mx.concatenate([window_kv, gathered], axis=1)
        else:
            kv_all = mx.concatenate([window_kv, compressed], axis=1)
    else:
        kv_all = window_kv

    sinks = self._sink_for(q.dtype)
    o = decode_attention(q, kv_all, sinks, self.scale)
    o = norm_rope(
        o.reshape(batch * self.n_heads, head_dim),
        None,
        None,
        -float(offset),
        freqs,
        rope_dims,
    ).reshape(batch, 1, self.n_heads * head_dim)
    o = self._grouped_output_projection(o)
    return self.wo_b(o)


def patch_deepseek_v4_decode_kernels() -> None:
    HyperConnection.hc_pre = _patched_hc_pre
    deepseek_v4._hc_expand_ops = _patched_hc_expand_ops  # pyright: ignore[reportPrivateUsage]
    V4Attention.__call__ = _patched_attention
