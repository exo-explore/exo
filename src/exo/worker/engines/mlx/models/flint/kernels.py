"""Fused Metal kernels for Flint's decoder layer.

Each kernel computes exactly what the MLX ops it replaces compute, bit for bit: the same
float32 arithmetic in the same order, with the same roundings. Fusing only cuts the number
of kernels per layer. A decode step is mostly small kernels, and each launch costs time on
its own, so that matters at batch 1.

- ``rms_norm_float32``: ``mx.fast.rms_norm(x.astype(f32), w.astype(f32), eps).astype(T)``,
  the float32 norm of the reference implementation, in one kernel instead of four. It sums
  the squares as MLX's ``rms_single_row`` kernel does: 4 reads per thread, ceil(D/4)
  threads rounded up to a simdgroup, then ``simd_sum`` across them.
- ``add_rms_norm_float32``: ``h = x + y`` (rounded to T, as ``mx.add``), then the norm of h.
- ``qk_norm_rope``: the query and key norms, then ``mx.fast.rope`` (traditional=False,
  scale 1) on both, written in the (B, heads, L, head_dim) layout attention reads.
- ``moe_combine``: ``(y * w[..., None]).sum(-2).astype(T) + shared``, optionally plus the
  residual stream, as the MoE block and the decoder layer compute them. MLX sums the k
  experts sequentially (``col_reduce_small``); so does the kernel, with no fused
  multiply-add, because MLX rounds the products before the sum.

- ``swiglu_down``: mlx-lm's ``swiglu(gate, up)`` (compiled: a bf16 sigmoid and two bf16
  multiplies) then a 4-bit down projection, computed as MLX's ``qmv_fast`` /
  ``gather_qmv_fast`` compute it: the same 16 values per thread, the same packed dot
  product, the same ``simd_sum``. MLX runs those kernels for one row (and the gather for
  every row, one token-expert pair each); for 2 to 12 rows it runs another kernel, so callers
  use this one only where MLX would run the same.

``fused_kernels_available()`` is False off Metal (CUDA, CPU), where callers keep the
original ops. The unit tests compare every kernel with those ops for bit equality.
"""

import math
from array import array
from functools import cache

import mlx.core as mx

# MLX's rms_single_row kernel covers rows up to this size; longer rows use another loop.
_MAX_ROW = 4096
_HEAD_DIM = 128

_RMS_SOURCE = """
    constexpr int N_READS = 4;
    constexpr int SIMD_SIZE = 32;
    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[SIMD_SIZE];
    uint gid = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint simd_lane_id = thread_index_in_simdgroup;
    uint simd_group_id = simdgroup_index_in_threadgroup;
    size_t row = size_t(gid) * D;

    float acc = 0;
    float thread_x[N_READS];
    LOAD
    acc = simd_sum(acc);
    if (simd_group_id == 0) {
      local_sums[simd_lane_id] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
      local_sums[simd_group_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
      acc = simd_sum(local_sums[simd_lane_id]);
      if (simd_lane_id == 0) {
        local_inv_mean[0] = metal::precise::rsqrt(acc / D + eps);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < N_READS; i++) {
      uint d = lid * N_READS + i;
      if (d < D) {
        float wi = static_cast<float>(w[d]);
        T normed = static_cast<T>(wi * (thread_x[i] * local_inv_mean[0]));
        out[row + d] = normed;
        STORE_FLOAT32
      }
    }
"""

_LOAD_X = """
    for (int i = 0; i < N_READS; i++) {
      uint d = lid * N_READS + i;
      thread_x[i] = d < D ? static_cast<float>(x[row + d]) : 0.0f;
      acc += thread_x[i] * thread_x[i];
    }
"""

_LOAD_X_PLUS_Y = """
    for (int i = 0; i < N_READS; i++) {
      uint d = lid * N_READS + i;
      if (d < D) {
        T hv = static_cast<T>(static_cast<float>(x[row + d]) + static_cast<float>(y[row + d]));
        h[row + d] = hv;
        thread_x[i] = static_cast<float>(hv);
      } else {
        thread_x[i] = 0.0f;
      }
      acc += thread_x[i] * thread_x[i];
    }
"""

# One simdgroup per (token, head) row of 128 values: thread t holds 4t..4t+3, as MLX's rms
# kernel reads a 128-value row, and the rotation partner of value j < 64 is j + 64, held
# by thread t ^ 16. mx.fast.rope takes the position as int + uint for array offsets (the
# sum is unsigned) and as a signed int for one offset at L = 1; so does this.
_QK_SOURCE = """
    constexpr int N_READS = 4;
    constexpr uint HALF_LANES = D / (2 * N_READS);
    uint row = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;
    uint h = row % (HQ + HK);
    uint n = row / (HQ + HK);
    uint b = n / L;
    uint l = n % L;
    bool is_q = h < HQ;
    const device T* src = is_q ? (q + size_t(n) * Q_STRIDE + h * D)
                               : (k + size_t(n) * K_STRIDE + K_OFFSET + (h - HQ) * D);
    const device T* w = is_q ? q_weight : k_weight;

    float v[N_READS];
    float acc = 0;
    for (int i = 0; i < N_READS; i++) {
      v[i] = static_cast<float>(src[lane * N_READS + i]);
      acc += v[i] * v[i];
    }
    acc = simd_sum(acc);
    float inv_mean = metal::precise::rsqrt(acc / D + eps);
    T y[N_READS];
    for (int i = 0; i < N_READS; i++) {
      float wi = static_cast<float>(w[lane * N_READS + i]);
      y[i] = static_cast<T>(wi * (v[i] * inv_mean));
    }
    if (ROPE) {
      float position;
      if (SIGNED_POSITION) {
        position = static_cast<float>(offset[0]);
      } else {
        position = static_cast<float>(l + offset[b * OFFSET_STRIDE]);
      }
      bool first_half = lane < HALF_LANES;
      for (int i = 0; i < N_READS; i++) {
        float self_value = static_cast<float>(y[i]);
        float other = simd_shuffle_xor(self_value, HALF_LANES);
        uint j = (lane % HALF_LANES) * N_READS + i;
        float d = static_cast<float>(j) / static_cast<float>(D / 2);
        float inv_freq = metal::exp2(-d * log2_base);
        float theta = position * inv_freq;
        float c = metal::fast::cos(theta);
        float s = metal::fast::sin(theta);
        float rotated;
        if (first_half) {
          rotated = self_value * c - other * s;
        } else {
          rotated = other * s + self_value * c;
        }
        y[i] = static_cast<T>(rotated);
      }
    }
    device T* dst = is_q ? (q_out + ((size_t(b) * HQ + h) * L + l) * D)
                         : (k_out + ((size_t(b) * HK + (h - HQ)) * L + l) * D);
    for (int i = 0; i < N_READS; i++) {
      dst[lane * N_READS + i] = y[i];
    }
"""

_COMBINE_SOURCE = """
    uint col = thread_position_in_grid.x;
    uint n = thread_position_in_grid.y;
    if (col >= D) {
      return;
    }
    auto expert_rows = y + size_t(n) * K * D + col;
    auto expert_weights = w + size_t(n) * K;
    float acc = 0.0f;
    for (int e = 0; e < K; e++) {
      float product = static_cast<float>(expert_rows[e * D]) * expert_weights[e];
      acc += product;
    }
    T routed = static_cast<T>(acc);
    T mixed = static_cast<T>(static_cast<float>(routed) + static_cast<float>(shared[size_t(n) * D + col]));
    if (RESIDUAL) {
      mixed = static_cast<T>(static_cast<float>(residual[size_t(n) * D + col]) + static_cast<float>(mixed));
    }
    out[size_t(n) * D + col] = mixed;
"""

# MLX's qmv_fast_impl (4 bits, group size GS, 16 values per thread, 4 rows per simdgroup, 2
# simdgroups), with its input row computed as swiglu(gate, up) from a gate/up row of 2K.
_SWIGLU_DOWN_SOURCE = """
    constexpr int packs_per_thread = 2;
    constexpr int num_simdgroups = 2;
    constexpr int results_per_simdgroup = 4;
    constexpr int pack_factor = 8;
    constexpr int bytes_per_pack = 4;
    constexpr int values_per_thread = pack_factor * packs_per_thread;
    constexpr int block_size = values_per_thread * 32;
    constexpr int scale_step_per_thread = GS / values_per_thread;
    typedef float U;

    uint3 tid = threadgroup_position_in_grid;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    thread U x_thread[values_per_thread];
    thread U result[results_per_simdgroup] = {0};

    const int in_vec_size_w = K * bytes_per_pack / pack_factor;
    const int in_vec_size_g = K / GS;
    const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
        simd_gid * results_per_simdgroup;

    size_t expert = GATHER ? size_t(indices[tid.x]) : 0;
    const device uint8_t* ws = (const device uint8_t*)w + expert * (size_t(N) * in_vec_size_w);
    const device T* sc = scales + expert * (size_t(N) * in_vec_size_g);
    const device T* bi = biases + expert * (size_t(N) * in_vec_size_g);
    ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
    sc += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    bi += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    const device T* g = gate_up + size_t(tid.x) * (2 * K) + simd_lid * values_per_thread;
    const device T* u = g + K;
    device T* yr = y + size_t(tid.x) * N + out_row;

    for (int k = 0; k < K; k += block_size) {
      thread T x[values_per_thread];
      for (int i = 0; i < values_per_thread; i++) {
        T gi = g[i];
        T silu = gi * flint_sigmoid(gi);
        x[i] = silu * u[i];
      }
      U sum = 0;
      for (int i = 0; i < values_per_thread; i += 4) {
        sum += U(x[i]) + U(x[i + 1]) + U(x[i + 2]) + U(x[i + 3]);
        x_thread[i] = x[i];
        x_thread[i + 1] = x[i + 1] / 16.0f;
        x_thread[i + 2] = x[i + 2] / 256.0f;
        x_thread[i + 3] = x[i + 3] / 4096.0f;
      }
      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint16_t*)(ws + row * in_vec_size_w);
        U s = sc[row * in_vec_size_g];
        U b = bi[row * in_vec_size_g];
        U accum = 0;
        for (int i = 0; i < (values_per_thread / 4); i++) {
          accum +=
              (x_thread[4 * i] * (wl[i] & 0x000f) +
               x_thread[4 * i + 1] * (wl[i] & 0x00f0) +
               x_thread[4 * i + 2] * (wl[i] & 0x0f00) +
               x_thread[4 * i + 3] * (wl[i] & 0xf000));
        }
        result[row] += s * accum + sum * b;
      }
      ws += block_size * bytes_per_pack / pack_factor;
      sc += block_size / GS;
      bi += block_size / GS;
      g += block_size;
      u += block_size;
    }
    for (int row = 0; row < results_per_simdgroup; row++) {
      result[row] = simd_sum(result[row]);
      if (simd_lid == 0) {
        yr[row] = static_cast<T>(result[row]);
      }
    }
"""

# MLX's Sigmoid (unary_ops.h), as the kernels mx.compile builds apply it to T.
_SIGMOID_HEADER = """
template <typename T>
T flint_sigmoid(T x) {
  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}
"""

# MLX rounds the expert products in their own kernel, before the sum: no fused multiply-add.
_NO_CONTRACT = "#pragma METAL fp contract(off)\n"


@cache
def _kernels() -> "dict[str, mx.fast.MetalKernel]":
    return {
        "rms": mx.fast.metal_kernel(
            name="flint_rms_norm_float32",
            input_names=["x", "w", "eps"],
            output_names=["out"],
            source=_RMS_SOURCE.replace("LOAD", _LOAD_X).replace("STORE_FLOAT32", ""),
        ),
        "add_rms": mx.fast.metal_kernel(
            name="flint_add_rms_norm_float32",
            input_names=["x", "y", "w", "eps"],
            output_names=["h", "out"],
            source=_RMS_SOURCE.replace("LOAD", _LOAD_X_PLUS_Y).replace(
                "STORE_FLOAT32", ""
            ),
        ),
        "add_rms_f32": mx.fast.metal_kernel(
            name="flint_add_rms_norm_float32_with_copy",
            input_names=["x", "y", "w", "eps"],
            output_names=["h", "out", "out_float32"],
            source=_RMS_SOURCE.replace("LOAD", _LOAD_X_PLUS_Y).replace(
                "STORE_FLOAT32", "out_float32[row + d] = static_cast<float>(normed);"
            ),
        ),
        "qk": mx.fast.metal_kernel(
            name="flint_qk_norm_rope",
            input_names=[
                "q",
                "k",
                "q_weight",
                "k_weight",
                "eps",
                "offset",
                "log2_base",
            ],
            output_names=["q_out", "k_out"],
            source=_QK_SOURCE,
        ),
        "combine": mx.fast.metal_kernel(
            name="flint_moe_combine",
            input_names=["y", "w", "shared", "residual"],
            output_names=["out"],
            source=_COMBINE_SOURCE,
            header=_NO_CONTRACT,
        ),
        "swiglu_down": mx.fast.metal_kernel(
            name="flint_swiglu_down_qmv",
            input_names=["gate_up", "w", "scales", "biases", "indices"],
            output_names=["y"],
            source=_SWIGLU_DOWN_SOURCE,
            header=_SIGMOID_HEADER,
        ),
    }


@cache
def _metal_available() -> bool:
    return mx.metal.is_available()


def fused_kernels_available() -> bool:
    """The kernels need Metal, and the default device to be the GPU."""
    return _metal_available() and mx.default_device().type == mx.gpu


@cache
def _float32_scalar(value: float) -> mx.array:
    scalar = mx.array(value, dtype=mx.float32)
    mx.eval(scalar)
    return scalar


@cache
def _log2_float32(base: float) -> float:
    """log2 of the float32 base, rounded to float32, as MLX's RoPE computes it on the host
    (``std::log2(float)``, correctly rounded for the bases the tests check)."""
    return array("f", [math.log2(array("f", [base])[0])])[0]


def _norm_threads(dim: int) -> int:
    return 32 * (((dim + 3) // 4 + 31) // 32)


def _reference_rms_norm_float32(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    return mx.fast.rms_norm(
        x.astype(mx.float32), weight.astype(mx.float32), eps
    ).astype(x.dtype)


def _norm_fusable(x: mx.array) -> bool:
    return (
        fused_kernels_available()
        and x.dtype in (mx.bfloat16, mx.float16)
        and x.shape[-1] <= _MAX_ROW
    )


def rms_norm_float32(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    """RMSNorm normalised and weighted in float32, rounded to x's dtype once."""
    if not _norm_fusable(x):
        return _reference_rms_norm_float32(x, weight, eps)
    dim = x.shape[-1]
    threads = _norm_threads(dim)
    return _kernels()["rms"](
        inputs=[x, weight, _float32_scalar(eps)],
        template=[("T", x.dtype), ("D", dim)],
        grid=(x.size // dim * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


def add_rms_norm_float32(
    x: mx.array, y: mx.array, weight: mx.array, eps: float
) -> tuple[mx.array, mx.array]:
    """``h = x + y`` and ``rms_norm_float32(h)``."""
    if not _norm_fusable(x) or x.shape != y.shape or x.dtype != y.dtype:
        h = x + y
        return h, _reference_rms_norm_float32(h, weight, eps)
    dim = x.shape[-1]
    threads = _norm_threads(dim)
    h, normed = _kernels()["add_rms"](
        inputs=[x, y, weight, _float32_scalar(eps)],
        template=[("T", x.dtype), ("D", dim)],
        grid=(x.size // dim * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[x.shape, x.shape],
        output_dtypes=[x.dtype, x.dtype],
    )
    return h, normed


def add_rms_norm_float32_with_copy(
    x: mx.array, y: mx.array, weight: mx.array, eps: float
) -> tuple[mx.array, mx.array, mx.array]:
    """``add_rms_norm_float32`` and its normed output cast to float32 (the MoE router's
    input, ``normed.astype(float32)``)."""
    if not _norm_fusable(x) or x.shape != y.shape or x.dtype != y.dtype:
        h, normed = add_rms_norm_float32(x, y, weight, eps)
        return h, normed, normed.astype(mx.float32)
    dim = x.shape[-1]
    threads = _norm_threads(dim)
    h, normed, normed_float32 = _kernels()["add_rms_f32"](
        inputs=[x, y, weight, _float32_scalar(eps)],
        template=[("T", x.dtype), ("D", dim)],
        grid=(x.size // dim * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[x.shape, x.shape, x.shape],
        output_dtypes=[x.dtype, x.dtype, mx.float32],
    )
    return h, normed, normed_float32


def qk_norm_rope_fusable(q: mx.array, head_dim: int) -> bool:
    return (
        fused_kernels_available()
        and q.dtype in (mx.bfloat16, mx.float16)
        and head_dim == _HEAD_DIM
    )


def qk_norm_rope(
    q: mx.array,
    k: mx.array,
    *,
    q_weight: mx.array,
    k_weight: mx.array,
    eps: float,
    n_heads: int,
    n_kv_heads: int,
    offset: int | mx.array,
    rope_base: float | None,
    k_column: int = 0,
) -> tuple[mx.array, mx.array]:
    """q (B, L, n_heads * 128) and k (B, L, n_kv_heads * 128), straight from the projections:
    each head normed in float32 (one rounding), then, unless rope_base is None, rotated as
    ``mx.fast.rope(traditional=False, base=rope_base, scale=1.0, offset=offset)``. Returns
    queries (B, n_heads, L, 128) and keys (B, n_kv_heads, L, 128). Check
    ``qk_norm_rope_fusable`` first.

    q and k may be one array, the output of a stacked q/k/v projection: then q is its first
    n_heads * 128 columns and k starts at column ``k_column``."""
    batch, length, _ = q.shape
    if isinstance(offset, mx.array):
        offsets = offset.reshape(-1).astype(mx.int32)
    else:
        offsets = mx.array([offset], dtype=mx.int32)
    rope = rope_base is not None
    signed_position = length == 1 and offsets.size == 1
    queries, keys = _kernels()["qk"](
        inputs=[
            q,
            k,
            q_weight,
            k_weight,
            _float32_scalar(eps),
            offsets,
            _float32_scalar(_log2_float32(rope_base) if rope_base is not None else 0.0),
        ],
        template=[
            ("T", q.dtype),
            ("D", _HEAD_DIM),
            ("HQ", n_heads),
            ("HK", n_kv_heads),
            ("L", length),
            ("ROPE", rope),
            ("SIGNED_POSITION", signed_position),
            ("OFFSET_STRIDE", int(offsets.size > 1)),
            ("Q_STRIDE", q.shape[-1]),
            ("K_STRIDE", k.shape[-1]),
            ("K_OFFSET", k_column),
        ],
        grid=(batch * length * (n_heads + n_kv_heads) * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[
            (batch, n_heads, length, _HEAD_DIM),
            (batch, n_kv_heads, length, _HEAD_DIM),
        ],
        output_dtypes=[q.dtype, q.dtype],
    )
    return queries, keys


def moe_combine(
    y: mx.array,
    weights: mx.array,
    shared: mx.array,
    residual: mx.array | None = None,
) -> mx.array:
    """``(y * weights[..., None]).sum(-2).astype(y.dtype) + shared`` (+ ``residual``), for
    the experts' outputs y (..., k, D), their float32 weights (..., k) and the shared
    expert's output (..., D)."""
    fusable = (
        fused_kernels_available()
        and y.dtype in (mx.bfloat16, mx.float16)
        and weights.dtype == mx.float32
        and shared.dtype == y.dtype
        and (residual is None or residual.dtype == y.dtype)
    )
    if not fusable:
        out = (y * weights[..., None]).sum(axis=-2).astype(y.dtype) + shared
        return out if residual is None else residual + out
    *lead, experts, dim = y.shape
    tokens = y.size // (experts * dim)
    return _kernels()["combine"](
        inputs=[y, weights, shared, residual if residual is not None else shared],
        template=[
            ("T", y.dtype),
            ("D", dim),
            ("K", experts),
            ("RESIDUAL", residual is not None),
        ],
        grid=(dim, tokens, 1),
        threadgroup=(min(256, dim), 1, 1),
        output_shapes=[(*lead, dim)],
        output_dtypes=[y.dtype],
    )[0]


@cache
def apple_gpu_generation() -> int | None:
    """The Apple GPU generation (13 = M1 ... 16 = M4, 17 = M5), None if unknown."""
    if not _metal_available():
        return None
    architecture = str(mx.device_info().get("architecture", ""))
    digits = "".join(c for c in architecture.removeprefix("applegpu_g") if c.isdigit())
    return int(digits) if digits else None


def swiglu_down_fusable(
    gate_up: mx.array,
    down: tuple[mx.array, mx.array, mx.array],
    group_size: int,
    bits: int,
) -> bool:
    """Whether ``swiglu_down`` computes what MLX's down projection would: the 4-bit
    matrix-vector kernel MLX runs for these shapes, on the GPUs it was checked on (M1 to
    M4)."""
    weight, scales, _ = down
    generation = apple_gpu_generation()
    k = gate_up.shape[-1] // 2
    return (
        fused_kernels_available()
        and generation is not None
        and generation <= 16
        and bits == 4
        and group_size == 64
        and gate_up.dtype in (mx.bfloat16, mx.float16)
        and scales.dtype == gate_up.dtype
        and k % 512 == 0
        and weight.shape[-1] * 8 == k
        and weight.shape[-2] % 8 == 0
    )


def swiglu_down(
    gate_up: mx.array,
    down: tuple[mx.array, mx.array, mx.array],
    indices: mx.array | None = None,
) -> mx.array:
    """``down(swiglu(gate, up))`` for rows of ``gate_up`` = [gate | up] (..., 2K), with the
    4-bit down projection's (weight, scales, biases) (N, K) or, with ``indices`` (one expert
    per row), (E, N, K). Returns (rows, N). Check ``swiglu_down_fusable`` first."""
    weight, scales, biases = down
    rows = gate_up.size // gate_up.shape[-1]
    k = gate_up.shape[-1] // 2
    n = weight.shape[-2]
    gather = indices is not None
    return _kernels()["swiglu_down"](
        inputs=[
            gate_up,
            weight,
            scales,
            biases,
            indices.reshape(-1) if indices is not None else _no_indices(),
        ],
        template=[
            ("T", gate_up.dtype),
            ("K", k),
            ("N", n),
            ("GS", 64),
            ("GATHER", gather),
        ],
        grid=(rows * 32, (n // 8) * 2, 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[gate_up.dtype],
    )[0]


@cache
def _no_indices() -> mx.array:
    indices = mx.zeros((1,), dtype=mx.uint32)
    mx.eval(indices)
    return indices
