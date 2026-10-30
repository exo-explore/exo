"""Fused Metal kernels for Laguna's decode and prefill.

Each kernel replaces a chain of small MLX ops with one dispatch and computes exactly what
that chain computes: the same float32 arithmetic in the same order, rounded to the model's
dtype at the same points, and every reduction in the order MLX's own kernel uses. They are
bit-identical to the ops they replace (the tests compare them bit for bit).

- ``qk_norm_rope``: the per-head RMSNorm of the queries and keys, YaRN's magnitude scale on
  the rotated part, and RoPE, read straight from the fused q/k/v projection output; it also
  lays the values out per head. Replaces 2 RMSNorms, 2 RoPEs and (on YaRN layers) 2 scaled
  slice updates, plus the copies a strided slice would need.
- ``route``: the router's sigmoid, score-correction bias, stable top-k and normalised
  weights. Replaces a cast, sigmoid, add, negate, full sort, gather, sum and divide.
- ``combine_experts``: the weighted sum of the routed experts, the routed scaling factor,
  the shared expert (on its own or stacked after the routed ones) and optionally the
  residual and the next layer's input RMSNorm. Replaces a multiply, sum, cast, scale, one or
  two adds and an RMSNorm.
- ``add_rms_norm``: a residual add followed by RMSNorm, returning both.
"""

import hashlib
import struct
from collections.abc import Callable
from functools import cache
from typing import cast

import mlx.core as mx

_SIMD = 32
# MLX's RMSNorm reads 4 values per thread (RMS_N_READS) and keeps rows of up to 4096 values
# in registers (RMS_LOOPED_LIMIT); the fused kernels follow the same layout.
_RMS_READS = 4
_RMS_LOOPED_LIMIT = 4096
# The longest row the fused RMSNorm kernels take (MLX's single-row RMSNorm limit).
MAX_ROW = _RMS_LOOPED_LIMIT

KernelFn = Callable[..., list[mx.array]]


def _kernel_name(prefix: str, source: str) -> str:
    """A kernel name unique to its source: MLX caches compiled kernels by name (and input
    types), and the sources bake in constants (shapes, epsilon, RoPE base, scales)."""
    return f"{prefix}_{hashlib.sha1(source.encode()).hexdigest()[:16]}"


def _float_literal(value: float) -> str:
    """An exact Metal float literal for a float32 value."""
    as_float32 = cast(float, struct.unpack("<f", struct.pack("<f", value))[0])
    return f"{as_float32.hex()}f"


@cache
def _qk_norm_rope_kernel(
    n_q_heads: int,
    n_kv_heads: int,
    head_dim: int,
    rotary_dims: int,
    yarn_freqs: bool,
    base_log2: float,
    eps: float,
    magnitude_scale: float | None,
    layout: tuple[int, int, int, int, int, int],
) -> KernelFn:
    """``layout``: (row width, first column) of the q, k and v sources: the fused
    projection's output three times, or three separate outputs."""
    assert head_dim == _SIMD * _RMS_READS, "one simdgroup per head"
    half = rotary_dims // 2
    assert half % _RMS_READS == 0 and rotary_dims <= head_dim
    partner = half // _RMS_READS
    q_row, q_col, k_row, k_col, v_row, v_col = layout
    if yarn_freqs:
        inverse_frequency = "float inv_freq = 1.0 / (freqs[pair]);"
    else:
        inverse_frequency = (
            f"float d = static_cast<float>(pair) / static_cast<float>({half});"
            f"float inv_freq = metal::exp2(-d * {_float_literal(base_log2)});"
        )
    scale_rotated = (
        ""
        if magnitude_scale is None
        else f"""
        if (dim < {rotary_dims}) {{
          y[i] = y[i] * static_cast<T>({_float_literal(magnitude_scale)});
        }}"""
    )
    source = f"""
    uint lane = thread_position_in_threadgroup.x;
    uint head = threadgroup_position_in_grid.x;
    uint row = threadgroup_position_in_grid.y;
    uint seq_len = q_src_shape[1];
    uint batch = row / seq_len;
    uint pos_in_seq = row - batch * seq_len;

    if (head >= {n_q_heads + n_kv_heads}) {{
      // Values: copy into [batch, head, position, dim].
      uint v_head = head - {n_q_heads + n_kv_heads};
      const device T* src = v_src + size_t(row) * {v_row} + {v_col} + v_head * {head_dim} + lane * {_RMS_READS};
      device T* dst = v_out + ((size_t(batch) * {n_kv_heads} + v_head) * seq_len + pos_in_seq) * {head_dim} + lane * {_RMS_READS};
      for (int i = 0; i < {_RMS_READS}; i++) {{
        dst[i] = src[i];
      }}
      return;
    }}

    bool is_q = head < {n_q_heads};
    const device T* w = is_q ? q_weight : k_weight;
    const device T* src = is_q
        ? q_src + size_t(row) * {q_row} + {q_col} + head * {head_dim}
        : k_src + size_t(row) * {k_row} + {k_col} + (head - {n_q_heads}) * {head_dim};
    src += lane * {_RMS_READS};

    // RMSNorm exactly as MLX's rms_single_row computes it.
    float acc = 0;
    float thread_x[{_RMS_READS}];
    for (int i = 0; i < {_RMS_READS}; i++) {{
      thread_x[i] = src[i];
      acc += thread_x[i] * thread_x[i];
    }}
    acc = simd_sum(acc);
    float inv_mean = metal::precise::rsqrt(acc / static_cast<uint>({head_dim}) + {_float_literal(eps)});
    T y[{_RMS_READS}];
    for (int i = 0; i < {_RMS_READS}; i++) {{
      uint dim = lane * {_RMS_READS} + i;
      y[i] = w[dim] * static_cast<T>(thread_x[i] * inv_mean);{scale_rotated}
    }}

    // RoPE (not traditional) exactly as MLX's rope kernel computes it: dims d and
    // d + half rotate together; dims past the rotary part pass through.
    int position = offsets[batch] + int(pos_in_seq);
    float L = 1.0f * static_cast<float>(position);
    T out_vals[{_RMS_READS}];
    for (int i = 0; i < {_RMS_READS}; i++) {{
      uint dim = lane * {_RMS_READS} + i;
      float mine = static_cast<float>(y[i]);
      float other = simd_shuffle_xor(mine, ushort({partner}));
      if (dim < {rotary_dims}) {{
        uint pair = dim < {half} ? dim : dim - {half};
        {inverse_frequency}
        float theta = L * inv_freq;
        float costheta = metal::fast::cos(theta);
        float sintheta = metal::fast::sin(theta);
        if (dim < {half}) {{
          float x1 = mine;
          float x2 = other;
          float rx1 = x1 * costheta - x2 * sintheta;
          out_vals[i] = static_cast<T>(rx1);
        }} else {{
          float x1 = other;
          float x2 = mine;
          float rx2 = x1 * sintheta + x2 * costheta;
          out_vals[i] = static_cast<T>(rx2);
        }}
      }} else {{
        out_vals[i] = y[i];
      }}
    }}

    device T* dst;
    if (is_q) {{
      dst = q_out + ((size_t(batch) * {n_q_heads} + head) * seq_len + pos_in_seq) * {head_dim};
    }} else {{
      dst = k_out + ((size_t(batch) * {n_kv_heads} + head - {n_q_heads}) * seq_len + pos_in_seq) * {head_dim};
    }}
    dst += lane * {_RMS_READS};
    for (int i = 0; i < {_RMS_READS}; i++) {{
      dst[i] = out_vals[i];
    }}
    """
    return cast(
        KernelFn,
        mx.fast.metal_kernel(
            name=_kernel_name("laguna_qk_norm_rope", source),
            input_names=[
                "q_src",
                "k_src",
                "v_src",
                "q_weight",
                "k_weight",
                "freqs",
                "offsets",
            ],
            output_names=["q_out", "k_out", "v_out"],
            source=source,
        ),
    )


@cache
def _no_freqs() -> mx.array:
    """The freqs input of a plain-RoPE layer (unused), made once rather than per call."""
    placeholder = mx.zeros((1,), dtype=mx.float32)
    mx.eval(placeholder)
    return placeholder


@cache
def _base_log2(base: float) -> float:
    """log2 of the RoPE base in float32, as MLX's rope passes it to its kernel."""
    return float(mx.log2(mx.array(base, dtype=mx.float32)).item())


def qk_norm_rope(
    sources: tuple[mx.array, mx.array, mx.array],
    columns: tuple[int, int, int],
    q_weight: mx.array,
    k_weight: mx.array,
    *,
    n_q_heads: int,
    n_kv_heads: int,
    head_dim: int,
    eps: float,
    rotary_dims: int,
    freqs: mx.array | None,
    base: float,
    magnitude_scale: float | None,
    offsets: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """The q, k and v projections -> queries and keys normalised and rotated, and values,
    each [B, heads, L, D]. ``sources`` are [B, L, width] arrays holding them from column
    ``columns`` on: the fused projection's output three times, or three separate outputs.
    ``offsets`` [B] int32 holds each row's first position; ``freqs`` are YaRN's frequencies
    (None for plain RoPE with ``base``)."""
    batch, seq_len, _ = sources[0].shape
    layout = (
        sources[0].shape[-1],
        columns[0],
        sources[1].shape[-1],
        columns[1],
        sources[2].shape[-1],
        columns[2],
    )
    kernel = _qk_norm_rope_kernel(
        n_q_heads,
        n_kv_heads,
        head_dim,
        rotary_dims,
        freqs is not None,
        _base_log2(base) if freqs is None else 0.0,
        eps,
        magnitude_scale,
        layout,
    )
    dtype = sources[0].dtype
    q, k, v = kernel(
        inputs=[
            *sources,
            q_weight,
            k_weight,
            freqs if freqs is not None else _no_freqs(),
            offsets,
        ],
        template=[("T", dtype)],
        grid=(_SIMD * (n_q_heads + 2 * n_kv_heads), batch * seq_len, 1),
        threadgroup=(_SIMD, 1, 1),
        output_shapes=[
            (batch, n_q_heads, seq_len, head_dim),
            (batch, n_kv_heads, seq_len, head_dim),
            (batch, n_kv_heads, seq_len, head_dim),
        ],
        output_dtypes=[dtype, dtype, dtype],
    )
    return q, k, v


@cache
def _route_kernel(n_experts: int, top_k: int, extra_index: int | None) -> KernelFn:
    """``extra_index``: an expert index appended after the top-k (the shared expert, when it
    is stacked with the routed ones)."""
    slots = top_k + int(extra_index is not None)
    append = (
        ""
        if extra_index is None
        else f"indices[size_t(row) * {slots} + {top_k}] = {extra_index};"
    )
    source = f"""
    uint expert = thread_position_in_threadgroup.x;
    uint row = threadgroup_position_in_grid.y;
    threadgroup float selection[{n_experts}];
    threadgroup float top_scores[{top_k}];
    threadgroup uint top_experts[{top_k}];

    // MLX's sigmoid on the float32 logits, then the score-correction bias.
    float x = static_cast<float>(logits[size_t(row) * {n_experts} + expert]);
    float y = 1 / (1 + metal::precise::exp(metal::abs(x)));
    float score = (x < 0) ? y : 1 - y;
    float selected = score + bias[expert];
    selection[expert] = selected;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // The top-k in MLX's argpartition order: MLX sorts (a stable merge sort), so the
    // experts come out by descending score, ties by ascending index.
    uint rank = 0;
    for (uint other = 0; other < {n_experts}; other++) {{
      float s = selection[other];
      rank += (s > selected) || (s == selected && other < expert);
    }}
    if (rank < {top_k}) {{
      top_scores[rank] = score;
      top_experts[rank] = expert;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (expert == 0) {{
      // weights / weights.sum(-1): MLX's small row reduction adds in order from 0.
      float total = 0;
      for (int r = 0; r < {top_k}; r++) {{
        total = top_scores[r] + total;
      }}
      for (int r = 0; r < {top_k}; r++) {{
        indices[size_t(row) * {slots} + r] = top_experts[r];
        weights[size_t(row) * {top_k} + r] = top_scores[r] / total;
      }}
      {append}
    }}
    """
    return cast(
        KernelFn,
        mx.fast.metal_kernel(
            name=_kernel_name("laguna_route", source),
            input_names=["logits", "bias"],
            output_names=["indices", "weights"],
            source=source,
        ),
    )


def route(
    logits: mx.array, bias: mx.array, top_k: int, extra_index: int | None = None
) -> tuple[mx.array, mx.array]:
    """Sigmoid routing with a score-correction bias: ``logits`` [..., E] -> (expert
    indices [..., k] uint32, normalised float32 weights [..., k]). With ``extra_index`` the
    indices get one more slot holding it."""
    *lead, n_experts = logits.shape
    rows = 1
    for size in lead:
        rows *= size
    slots = top_k + int(extra_index is not None)
    indices, weights = _route_kernel(n_experts, top_k, extra_index)(
        inputs=[logits, bias],
        grid=(n_experts, rows, 1),
        threadgroup=(n_experts, 1, 1),
        output_shapes=[(*lead, slots), (*lead, top_k)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return indices, weights


_COMBINE_HEADER = """
// One output value of the expert combination: sum_e w[e] * y[e] (MLX's col_reduce_small:
// min(8, k) partial sums, each adding every 8th expert in order from 0, then added in order),
// rounded, times the routed scaling factor, plus the shared expert, plus the residual.
template <typename T, typename YP, typename WP, typename SP, typename RP>
inline T laguna_combine_value(YP y, WP w, SP shared, RP resid, int top_k, int lanes,
                              size_t stride, T scale, bool has_resid) {
  float partial[8];
  for (int l = 0; l < lanes; l++) {
    partial[l] = 0;
  }
  for (int e = 0; e < top_k; e++) {
    float product = static_cast<float>(y[e * stride]) * w[e];
    partial[e % lanes] = product + partial[e % lanes];
  }
  float total = partial[0];
  for (int l = 1; l < lanes; l++) {
    total = partial[l] + total;
  }
  T routed = static_cast<T>(total);
  T out_v = routed * scale;
  out_v = out_v + shared[0];
  if (has_resid) {
    out_v = resid[0] + out_v;
  }
  return out_v;
}
"""


@cache
def _combine_kernel(
    top_k: int,
    slots: int,
    dim: int,
    scale: float,
    shared_in_slot: bool,
    residual: bool,
    norm_eps: float | None,
) -> KernelFn:
    """``slots`` experts per row in ``experts``: the top-k routed ones, then (when
    ``shared_in_slot``) the shared expert. With ``norm_eps`` it also returns the RMSNorm of
    the result (the next layer's input norm), computed as MLX's rms_single_row does."""
    lanes = min(8, top_k)
    shared_ptr = (
        f"experts + size_t(row) * {slots * dim} + {top_k * dim} + col"
        if shared_in_slot
        else f"shared + size_t(row) * {dim} + col"
    )
    resid_ptr = f"resid + size_t(row) * {dim} + col" if residual else shared_ptr
    value = f"""laguna_combine_value<T>(
          experts + size_t(row) * {slots * dim} + col,
          weights + size_t(row) * {top_k},
          {shared_ptr},
          {resid_ptr},
          {top_k}, {lanes}, {dim}, static_cast<T>({_float_literal(scale)}), {str(residual).lower()})"""
    if norm_eps is None:
        source = f"""
    uint col = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;
    out[size_t(row) * {dim} + col] = {value};
    """
        outputs = ["out"]
    else:
        assert dim <= _RMS_LOOPED_LIMIT and dim % _RMS_READS == 0
        source = f"""
    uint lid = thread_position_in_threadgroup.x;
    uint row = threadgroup_position_in_grid.y;
    uint simd_lane_id = lid % {_SIMD};
    uint simd_group_id = lid / {_SIMD};
    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[{_SIMD}];
    float acc = 0;
    float thread_x[{_RMS_READS}];
    bool in_row = lid * {_RMS_READS} + {_RMS_READS} <= {dim};
    for (int i = 0; i < {_RMS_READS}; i++) {{
      thread_x[i] = 0;
      if (in_row) {{
        uint col = lid * {_RMS_READS} + i;
        T v = {value};
        out[size_t(row) * {dim} + col] = v;
        thread_x[i] = v;
      }}
      acc += thread_x[i] * thread_x[i];
    }}
    // RMSNorm exactly as MLX's rms_single_row computes it.
    acc = simd_sum(acc);
    if (simd_group_id == 0) {{
      local_sums[simd_lane_id] = 0;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {{
      local_sums[simd_group_id] = acc;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {{
      acc = simd_sum(local_sums[simd_lane_id]);
      if (simd_lane_id == 0) {{
        local_inv_mean[0] = metal::precise::rsqrt(acc / static_cast<uint>({dim}) + {_float_literal(norm_eps)});
      }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (in_row) {{
      for (int i = 0; i < {_RMS_READS}; i++) {{
        uint col = lid * {_RMS_READS} + i;
        normed[size_t(row) * {dim} + col] = norm_weight[col] * static_cast<T>(thread_x[i] * local_inv_mean[0]);
      }}
    }}
    """
        outputs = ["out", "normed"]
    inputs = ["experts", "weights"]
    if not shared_in_slot:
        inputs.append("shared")
    if residual:
        inputs.append("resid")
    if norm_eps is not None:
        inputs.append("norm_weight")
    return cast(
        KernelFn,
        mx.fast.metal_kernel(
            name=_kernel_name("laguna_combine", source),
            input_names=inputs,
            output_names=outputs,
            source=source,
            header=_COMBINE_HEADER,
        ),
    )


def combine_experts(
    experts: mx.array,
    weights: mx.array,
    shared: mx.array | None,
    scale: float,
    residual: mx.array | None,
    norm: tuple[mx.array, float] | None = None,
) -> tuple[mx.array, mx.array | None]:
    """``((experts[..., :k, :] * weights[..., None]).sum(-2).astype(dtype) * scale +
    shared)``, plus ``residual`` when given. ``experts`` [..., k, D] with ``shared`` [..., D],
    or [..., k + 1, D] with the shared expert's output last (``shared`` None); ``weights``
    [..., k] float32. With ``norm`` (weight, eps) also returns rms_norm of the result."""
    *lead, slots, dim = experts.shape
    top_k = weights.shape[-1]
    shared_in_slot = shared is None
    assert slots == top_k + int(shared_in_slot)
    rows = 1
    for size in lead:
        rows *= size
    inputs = [experts, weights]
    if shared is not None:
        inputs.append(shared)
    if residual is not None:
        inputs.append(residual)
    if norm is not None:
        inputs.append(norm[0])
    kernel = _combine_kernel(
        top_k,
        slots,
        dim,
        scale,
        shared_in_slot,
        residual is not None,
        None if norm is None else norm[1],
    )
    dtype = experts.dtype
    if norm is None:
        (out,) = kernel(
            inputs=inputs,
            template=[("T", dtype)],
            grid=(dim, rows, 1),
            threadgroup=(min(dim, 256), 1, 1),
            output_shapes=[(*lead, dim)],
            output_dtypes=[dtype],
        )
        return out, None
    threads = _SIMD * ((dim // _RMS_READS + _SIMD - 1) // _SIMD)
    out, normed = kernel(
        inputs=inputs,
        template=[("T", dtype)],
        grid=(threads, rows, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(*lead, dim), (*lead, dim)],
        output_dtypes=[dtype, dtype],
    )
    return out, normed


@cache
def _add_rms_norm_kernel(dim: int, eps: float) -> KernelFn:
    assert dim <= _RMS_LOOPED_LIMIT and dim % _RMS_READS == 0
    threads = _SIMD * ((dim // _RMS_READS + _SIMD - 1) // _SIMD)
    source = f"""
    uint lid = thread_position_in_threadgroup.x;
    uint row = threadgroup_position_in_grid.y;
    uint simd_lane_id = lid % {_SIMD};
    uint simd_group_id = lid / {_SIMD};
    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[{_SIMD}];

    size_t base = size_t(row) * {dim} + lid * {_RMS_READS};
    float acc = 0;
    float thread_x[{_RMS_READS}];
    if (lid * {_RMS_READS} + {_RMS_READS} <= {dim}) {{
      for (int i = 0; i < {_RMS_READS}; i++) {{
        T h = x[base + i] + r[base + i];
        h_out[base + i] = h;
        thread_x[i] = h;
        acc += thread_x[i] * thread_x[i];
      }}
    }} else {{
      for (int i = 0; i < {_RMS_READS}; i++) {{
        thread_x[i] = 0;
        acc += thread_x[i] * thread_x[i];
      }}
    }}
    // RMSNorm exactly as MLX's rms_single_row computes it.
    acc = simd_sum(acc);
    if (simd_group_id == 0) {{
      local_sums[simd_lane_id] = 0;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {{
      local_sums[simd_group_id] = acc;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {{
      acc = simd_sum(local_sums[simd_lane_id]);
      if (simd_lane_id == 0) {{
        local_inv_mean[0] = metal::precise::rsqrt(acc / static_cast<uint>({dim}) + {_float_literal(eps)});
      }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid * {_RMS_READS} + {_RMS_READS} <= {dim}) {{
      for (int i = 0; i < {_RMS_READS}; i++) {{
        n_out[base + i] = w[lid * {_RMS_READS} + i] * static_cast<T>(thread_x[i] * local_inv_mean[0]);
      }}
    }}
    """
    del threads
    return cast(
        KernelFn,
        mx.fast.metal_kernel(
            name=_kernel_name("laguna_add_rms_norm", source),
            input_names=["x", "r", "w"],
            output_names=["h_out", "n_out"],
            source=source,
        ),
    )


def add_rms_norm(
    x: mx.array, r: mx.array, weight: mx.array, eps: float
) -> tuple[mx.array, mx.array]:
    """``h = x + r`` and ``rms_norm(h, weight, eps)``."""
    *lead, dim = x.shape
    rows = 1
    for size in lead:
        rows *= size
    threads = _SIMD * ((dim // _RMS_READS + _SIMD - 1) // _SIMD)
    h, n = _add_rms_norm_kernel(dim, eps)(
        inputs=[x, r, weight],
        template=[("T", x.dtype)],
        grid=(threads, rows, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[x.shape, x.shape],
        output_dtypes=[x.dtype, x.dtype],
    )
    return h, n
