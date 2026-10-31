"""Fused Metal kernels for Hy3's decoder layer.

Each kernel computes exactly what the MLX ops it replaces compute, bit for bit: the same
float32 arithmetic in the same order, with the same roundings. Fusing only cuts the number
of kernels per layer. A decode step is mostly small kernels, each waiting for the one
before it, so their count sets much of a step's time at batch 1.

- ``add_rms_norm``: ``h = x + y`` (rounded to T, as ``mx.add``), then ``nn.RMSNorm`` of h
  (``mx.fast.rms_norm``: normalised in float32, rounded to T, then multiplied by the T
  weight). It sums the squares as MLX's ``rms_single_row`` kernel does: 4 values per
  thread, ceil(D/4) threads rounded up to a simdgroup, ``simd_sum`` within and across
  simdgroups. Optionally also writes the normed output widened to float32 (the router's
  input, ``normed.astype(float32)``: exact).
- ``qk_norm_rope``: the per-head query and key RMSNorms, then ``mx.fast.rope``
  (traditional=False, scale 1) on both, written in the (B, heads, L, head_dim) layout
  attention reads, with the values copied into that layout too.
- ``route``: the router after its logits: the sigmoid (MLX's ``Sigmoid``), the expert bias,
  the top k in the order ``mx.argpartition`` returns them on the GPU (MLX's merge sort, a
  stable ascending sort of the negated scores: highest first, ties to the lower index; the
  kernel ranks every expert against all the others),
  their unbiased scores summed left to right from 0 (``row_reduce_small``), divided by the
  sum plus 1e-20 and multiplied by the scaling factor.
- ``moe_combine``: ``(y * w[..., None]).sum(-2)`` over the k experts, rounded to T, plus
  the shared expert (and the residual stream), as the MoE block and the decoder layer
  compute them. For k <= 8 MLX's ``col_reduce_small`` gives each expert its own thread row
  and adds them in order onto 0; the kernel does the same, with no fused multiply-add,
  because MLX rounds the products in their own kernel before the sum.
- ``moe_combine_norm``: ``moe_combine`` with the residual, then the next decoder layer's
  input RMSNorm of the result, in one kernel (the norm part as ``add_rms_norm``'s).
- ``swiglu_down``: mlx-lm's ``swiglu(gate, up)`` (compiled: a T sigmoid and two T
  multiplies) then a 4-bit down projection, computed as MLX's ``qmv_fast`` /
  ``gather_qmv_fast`` compute it: the same 16 values per thread, the same packed dot
  product, the same ``simd_sum``. MLX runs those kernels for one row (and the gather for
  every row, one token-expert pair each); for several rows of a plain matmul it runs
  another kernel, so callers use this one only where MLX would run the same.

The kernels follow the approach of exo's fused kernels for Flint. ``fused_kernels_available()``
is False off Metal (CUDA, CPU), where callers keep the original ops. The unit tests compare
every kernel with those ops for bit equality.
"""

import math
from array import array
from functools import cache

import mlx.core as mx

# MLX's rms_single_row kernel covers rows up to this size; longer rows use another loop.
_MAX_ROW = 4096
_HEAD_DIM = 128
# The router kernel runs one thread per expert in one threadgroup.
_MAX_EXPERTS = 1024
_MAX_TOP_K = 16

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
        T normed = w[d] * static_cast<T>(thread_x[i] * local_inv_mean[0]);
        out[row + d] = normed;
        if (FLOAT32_COPY) {
          out_float32[row + d] = static_cast<float>(normed);
        }
      }
    }
"""

# One simdgroup per (token, head) row of 128 values: thread t holds 4t..4t+3, as MLX's rms
# kernel reads a 128-value row (its second simd_sum adds zeros: the same sum), and the
# rotation partner of value j < 64 is j + 64, held by thread t ^ 16. mx.fast.rope takes the
# position as int + uint for array offsets (the sum is unsigned) and as a signed int for one
# offset at L = 1; so does this.
_QK_SOURCE = """
    constexpr int N_READS = 4;
    constexpr uint HALF_LANES = D / (2 * N_READS);
    uint row = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;
    uint h = row % (HQ + 2 * HK);
    uint n = row / (HQ + 2 * HK);
    uint b = n / L;
    uint l = n % L;
    if (h >= HQ + HK) {
      // values: copied into the (B, heads, L, head_dim) layout
      const device T* vs = v + size_t(n) * V_STRIDE + V_OFFSET + (h - HQ - HK) * D;
      device T* vd = v_out + ((size_t(b) * HK + (h - HQ - HK)) * L + l) * D;
      for (int i = 0; i < N_READS; i++) {
        vd[lane * N_READS + i] = vs[lane * N_READS + i];
      }
      return;
    }
    bool is_q = h < HQ;
    const device T* src = is_q ? (q + size_t(n) * Q_STRIDE + h * D)
                               : (k + size_t(n) * K_STRIDE + K_OFFSET + (h - HQ) * D);
    const device T* w = is_q ? q_weight : k_weight;

    float vals[N_READS];
    float acc = 0;
    for (int i = 0; i < N_READS; i++) {
      vals[i] = static_cast<float>(src[lane * N_READS + i]);
      acc += vals[i] * vals[i];
    }
    acc = simd_sum(acc);
    float inv_mean = metal::precise::rsqrt(acc / D + eps);
    T y[N_READS];
    for (int i = 0; i < N_READS; i++) {
      y[i] = w[lane * N_READS + i] * static_cast<T>(vals[i] * inv_mean);
    }
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
    device T* dst = is_q ? (q_out + ((size_t(b) * HQ + h) * L + l) * D)
                         : (k_out + ((size_t(b) * HK + (h - HQ)) * L + l) * D);
    for (int i = 0; i < N_READS; i++) {
      dst[lane * N_READS + i] = y[i];
    }
"""

# One threadgroup per token, one thread per expert (E threads, rounded up to a simdgroup).
# Each thread computes its expert's score and biased score, then its rank: how many experts
# come before it in a stable descending sort of the biased scores (a higher score, or the
# same score and a lower index). The first TOP_K ranks are argpartition's first TOP_K
# columns, in its order (MLX's merge sort, stable, of the negated scores).
_ROUTE_SOURCE = """
    uint n = threadgroup_position_in_grid.x;
    uint e = thread_position_in_threadgroup.x;
    threadgroup float biased[E];
    threadgroup float chosen[TOP_K];
    const device float* row = logits + size_t(n) * E;
    float score = 0.0f;
    float mine = 0.0f;
    if (e < E) {
      float x = row[e];
      float y = 1 / (1 + metal::precise::exp(metal::abs(x)));
      score = (x < 0) ? y : 1 - y;
      mine = score + bias[e];
      biased[e] = mine;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e < E) {
      uint rank = 0;
      for (uint j = 0; j < E; j++) {
        float other = biased[j];
        rank += (other > mine || (other == mine && j < e)) ? 1 : 0;
      }
      if (rank < TOP_K) {
        indices[size_t(n) * TOP_K + rank] = e;
        chosen[rank] = score;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e == 0) {
      float total = 0.0f;
      for (int k = 0; k < TOP_K; k++) {
        total = chosen[k] + total;
      }
      float denominator = total + 1e-20f;
      for (int k = 0; k < TOP_K; k++) {
        float weight = chosen[k] / denominator;
        weights[size_t(n) * TOP_K + k] = weight * scaling;
      }
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
    T shared_value = shared[size_t(n) * D + col];
    T mixed;
    if (FLOAT32_COMBINE) {
      mixed = static_cast<T>(acc + static_cast<float>(shared_value));
    } else {
      T routed = static_cast<T>(acc);
      mixed = static_cast<T>(static_cast<float>(routed) + static_cast<float>(shared_value));
    }
    if (RESIDUAL) {
      mixed = static_cast<T>(static_cast<float>(residual[size_t(n) * D + col]) + static_cast<float>(mixed));
    }
    out[size_t(n) * D + col] = mixed;
"""

# moe_combine with the residual, then the next layer's input RMSNorm of the result: one
# threadgroup per token, laid out as rms_single_row (4 columns per thread).
_COMBINE_NORM_SOURCE = """
    constexpr int N_READS = 4;
    constexpr int SIMD_SIZE = 32;
    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[SIMD_SIZE];
    uint gid = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint simd_lane_id = thread_index_in_simdgroup;
    uint simd_group_id = simdgroup_index_in_threadgroup;
    size_t row = size_t(gid) * D;
    auto expert_weights = w + size_t(gid) * K;

    float acc = 0;
    float thread_x[N_READS];
    for (int i = 0; i < N_READS; i++) {
      uint d = lid * N_READS + i;
      if (d < D) {
        auto expert_rows = y + size_t(gid) * K * D + d;
        float routed_sum = 0.0f;
        for (int e = 0; e < K; e++) {
          float product = static_cast<float>(expert_rows[e * D]) * expert_weights[e];
          routed_sum += product;
        }
        T shared_value = shared[row + d];
        T mixed;
        if (FLOAT32_COMBINE) {
          mixed = static_cast<T>(routed_sum + static_cast<float>(shared_value));
        } else {
          T routed = static_cast<T>(routed_sum);
          mixed = static_cast<T>(static_cast<float>(routed) + static_cast<float>(shared_value));
        }
        T hv = static_cast<T>(static_cast<float>(residual[row + d]) + static_cast<float>(mixed));
        h[row + d] = hv;
        thread_x[i] = static_cast<float>(hv);
      } else {
        thread_x[i] = 0.0f;
      }
      acc += thread_x[i] * thread_x[i];
    }
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
        out[row + d] = norm_weight[d] * static_cast<T>(thread_x[i] * local_inv_mean[0]);
      }
    }
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
    const device uint8_t* ws = (const device uint8_t*)w + expert * W_EXPERT_STRIDE;
    const device T* sc = scales + expert * S_EXPERT_STRIDE;
    const device T* bi = biases + expert * S_EXPERT_STRIDE;
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
        T silu = gi * hy3_sigmoid(gi);
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
T hy3_sigmoid(T x) {
  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}
"""

# MLX rounds products and sums in separate kernels: no fused multiply-add.
_NO_CONTRACT = "#pragma METAL fp contract(off)\n"


@cache
def _kernels() -> "dict[str, mx.fast.MetalKernel]":
    return {
        "add_rms": mx.fast.metal_kernel(
            name="hy3_add_rms_norm",
            input_names=["x", "y", "w", "eps"],
            output_names=["h", "out"],
            source=_RMS_SOURCE.replace("FLOAT32_COPY", "false").replace(
                "out_float32[row + d] = static_cast<float>(normed);", ""
            ),
        ),
        "add_rms_f32": mx.fast.metal_kernel(
            name="hy3_add_rms_norm_with_float32",
            input_names=["x", "y", "w", "eps"],
            output_names=["h", "out", "out_float32"],
            source=_RMS_SOURCE.replace("FLOAT32_COPY", "true"),
        ),
        "qk": mx.fast.metal_kernel(
            name="hy3_qk_norm_rope",
            input_names=[
                "q",
                "k",
                "v",
                "q_weight",
                "k_weight",
                "eps",
                "offset",
                "log2_base",
            ],
            output_names=["q_out", "k_out", "v_out"],
            source=_QK_SOURCE,
        ),
        "route": mx.fast.metal_kernel(
            name="hy3_route",
            input_names=["logits", "bias", "scaling"],
            output_names=["indices", "weights"],
            source=_ROUTE_SOURCE,
            header=_NO_CONTRACT,
        ),
        "combine": mx.fast.metal_kernel(
            name="hy3_moe_combine",
            input_names=["y", "w", "shared", "residual"],
            output_names=["out"],
            source=_COMBINE_SOURCE,
            header=_NO_CONTRACT,
        ),
        "combine_norm": mx.fast.metal_kernel(
            name="hy3_moe_combine_norm",
            input_names=["y", "w", "shared", "residual", "norm_weight", "eps"],
            output_names=["h", "out"],
            source=_COMBINE_NORM_SOURCE,
            header=_NO_CONTRACT,
        ),
        "swiglu_down": mx.fast.metal_kernel(
            name="hy3_swiglu_down_qmv",
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


def add_rms_norm(
    x: mx.array,
    y: mx.array,
    weight: mx.array,
    eps: float,
    *,
    float32_copy: bool = False,
) -> tuple[mx.array, mx.array, mx.array | None]:
    """``h = x + y``, ``normed = mx.fast.rms_norm(h, weight, eps)`` and, with
    ``float32_copy``, ``normed.astype(float32)``."""
    fusable = (
        fused_kernels_available()
        and x.dtype in (mx.bfloat16, mx.float16)
        and x.shape == y.shape
        and x.dtype == y.dtype == weight.dtype
        and weight.ndim == 1
        and x.shape[-1] == weight.shape[0]
        and x.shape[-1] <= _MAX_ROW
    )
    if not fusable:
        h = x + y
        normed = mx.fast.rms_norm(h, weight, eps)
        return h, normed, normed.astype(mx.float32) if float32_copy else None
    dim = x.shape[-1]
    threads = _norm_threads(dim)
    if float32_copy:
        h, normed, normed_float32 = _kernels()["add_rms_f32"](
            inputs=[x, y, weight, _float32_scalar(eps)],
            template=[("T", x.dtype), ("D", dim)],
            grid=(x.size // dim * threads, 1, 1),
            threadgroup=(threads, 1, 1),
            output_shapes=[x.shape, x.shape, x.shape],
            output_dtypes=[x.dtype, x.dtype, mx.float32],
        )
        return h, normed, normed_float32
    h, normed = _kernels()["add_rms"](
        inputs=[x, y, weight, _float32_scalar(eps)],
        template=[("T", x.dtype), ("D", dim)],
        grid=(x.size // dim * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[x.shape, x.shape],
        output_dtypes=[x.dtype, x.dtype],
    )
    return h, normed, None


def qk_norm_rope_fusable(q: mx.array, weight: mx.array, head_dim: int) -> bool:
    return (
        fused_kernels_available()
        and q.dtype in (mx.bfloat16, mx.float16)
        and weight.dtype == q.dtype
        and head_dim == _HEAD_DIM
    )


def qk_norm_rope(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    q_weight: mx.array,
    k_weight: mx.array,
    eps: float,
    n_heads: int,
    n_kv_heads: int,
    offset: int | mx.array,
    rope_base: float,
    k_column: int = 0,
    v_column: int = 0,
) -> tuple[mx.array, mx.array, mx.array]:
    """q (B, L, n_heads * 128), k and v (B, L, n_kv_heads * 128), straight from the
    projections: each query and key head through ``mx.fast.rms_norm(head, weight, eps)``,
    then ``mx.fast.rope(traditional=False, base=rope_base, scale=1.0, offset=offset)``.
    Returns queries (B, n_heads, L, 128), keys and values (B, n_kv_heads, L, 128), the
    values unchanged. Check ``qk_norm_rope_fusable`` first.

    q, k and v may be one array, the output of a stacked q/k/v projection: then q is its
    first n_heads * 128 columns, k starts at column ``k_column`` and v at ``v_column``."""
    batch, length, _ = q.shape
    if isinstance(offset, mx.array):
        offsets = offset.reshape(-1).astype(mx.int32)
    else:
        offsets = mx.array([offset], dtype=mx.int32)
    signed_position = length == 1 and offsets.size == 1
    queries, keys, values = _kernels()["qk"](
        inputs=[
            q,
            k,
            v,
            q_weight,
            k_weight,
            _float32_scalar(eps),
            offsets,
            _float32_scalar(_log2_float32(rope_base)),
        ],
        template=[
            ("T", q.dtype),
            ("D", _HEAD_DIM),
            ("HQ", n_heads),
            ("HK", n_kv_heads),
            ("L", length),
            ("SIGNED_POSITION", signed_position),
            ("OFFSET_STRIDE", int(offsets.size > 1)),
            ("Q_STRIDE", q.shape[-1]),
            ("K_STRIDE", k.shape[-1]),
            ("K_OFFSET", k_column),
            ("V_STRIDE", v.shape[-1]),
            ("V_OFFSET", v_column),
        ],
        grid=(batch * length * (n_heads + 2 * n_kv_heads) * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[
            (batch, n_heads, length, _HEAD_DIM),
            (batch, n_kv_heads, length, _HEAD_DIM),
            (batch, n_kv_heads, length, _HEAD_DIM),
        ],
        output_dtypes=[q.dtype, q.dtype, q.dtype],
    )
    return queries, keys, values


def route_fusable(logits: mx.array, top_k: int) -> bool:
    return (
        fused_kernels_available()
        and logits.dtype == mx.float32
        and logits.shape[-1] <= _MAX_EXPERTS
        and 0 < top_k <= min(_MAX_TOP_K, logits.shape[-1])
    )


def route(
    logits: mx.array,
    bias: mx.array,
    top_k: int,
    scaling: mx.array,
) -> tuple[mx.array, mx.array]:
    """The chosen experts' indices (uint32, highest biased score first) and their weights
    (float32), for float32 router logits (..., E), the float32 expert bias (E,) and the
    scaling factor as a float32 scalar. Check ``route_fusable`` first."""
    experts = logits.shape[-1]
    *lead, _ = logits.shape
    tokens = logits.size // experts
    threads = 32 * ((experts + 31) // 32)
    indices, weights = _kernels()["route"](
        inputs=[logits, bias, scaling],
        template=[("E", experts), ("TOP_K", top_k)],
        grid=(tokens * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(*lead, top_k), (*lead, top_k)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return indices, weights


def moe_combine(
    y: mx.array,
    weights: mx.array,
    shared: mx.array,
    residual: mx.array | None = None,
    *,
    float32_combine: bool = False,
) -> mx.array:
    """The MoE block's output, plus ``residual`` when given: for the experts' outputs
    y (..., k, D) and their float32 weights (..., k), ``(y * weights[..., None]).sum(-2)``
    rounded to T plus the shared expert's output (or, with ``float32_combine``, the sum plus
    the shared output in float32, rounded once)."""
    fusable = (
        fused_kernels_available()
        and y.dtype in (mx.bfloat16, mx.float16)
        and weights.dtype == mx.float32
        and weights.shape[-1] == y.shape[-2]
        and y.shape[-2] <= 8
        and shared.dtype == y.dtype
        and (residual is None or residual.dtype == y.dtype)
    )
    if not fusable:
        routed = (y * weights[..., None]).sum(axis=-2)
        if float32_combine:
            out = (routed + shared.astype(mx.float32)).astype(y.dtype)
        else:
            out = routed.astype(y.dtype) + shared
        return out if residual is None else residual + out
    *lead, experts, dim = y.shape
    tokens = y.size // (experts * dim)
    return _kernels()["combine"](
        inputs=[y, weights, shared, residual if residual is not None else shared],
        template=[
            ("T", y.dtype),
            ("D", dim),
            ("K", experts),
            ("FLOAT32_COMBINE", float32_combine),
            ("RESIDUAL", residual is not None),
        ],
        grid=(dim, tokens, 1),
        threadgroup=(min(256, dim), 1, 1),
        output_shapes=[(*lead, dim)],
        output_dtypes=[y.dtype],
    )[0]


def moe_combine_norm(
    y: mx.array,
    weights: mx.array,
    shared: mx.array,
    residual: mx.array,
    norm_weight: mx.array,
    eps: float,
    *,
    float32_combine: bool = False,
) -> tuple[mx.array, mx.array]:
    """``h = moe_combine(y, weights, shared, residual)`` and ``mx.fast.rms_norm(h,
    norm_weight, eps)`` (the next decoder layer's input norm)."""
    fusable = (
        fused_kernels_available()
        and y.dtype in (mx.bfloat16, mx.float16)
        and weights.dtype == mx.float32
        and weights.shape[-1] == y.shape[-2]
        and y.shape[-2] <= 8
        and shared.dtype == residual.dtype == norm_weight.dtype == y.dtype
        and norm_weight.ndim == 1
        and norm_weight.shape[0] == y.shape[-1]
        and y.shape[-1] <= _MAX_ROW
    )
    if not fusable:
        h = moe_combine(y, weights, shared, residual, float32_combine=float32_combine)
        return h, mx.fast.rms_norm(h, norm_weight, eps)
    *lead, experts, dim = y.shape
    tokens = y.size // (experts * dim)
    threads = _norm_threads(dim)
    h, normed = _kernels()["combine_norm"](
        inputs=[y, weights, shared, residual, norm_weight, _float32_scalar(eps)],
        template=[
            ("T", y.dtype),
            ("D", dim),
            ("K", experts),
            ("FLOAT32_COMBINE", float32_combine),
        ],
        grid=(tokens * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(*lead, dim), (*lead, dim)],
        output_dtypes=[y.dtype, y.dtype],
    )
    return h, normed


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
    weight, scales, biases = down
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
        and biases.dtype == gate_up.dtype
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
    per row), (E, N, K) (a prefix of a larger stack of experts works too). Returns
    (rows, N). Check ``swiglu_down_fusable`` first."""
    weight, scales, biases = down
    rows = gate_up.size // gate_up.shape[-1]
    k = gate_up.shape[-1] // 2
    n = weight.shape[-2]
    gather = indices is not None
    # Experts are contiguous (a prefix of a larger stack is too): the stride is one
    # expert's (N, K) block, in bytes for the packed weight and in elements for scales.
    weight_stride = n * weight.shape[-1] * 4 if gather else 0
    scale_stride = n * scales.shape[-1] if gather else 0
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
            ("W_EXPERT_STRIDE", weight_stride),
            ("S_EXPERT_STRIDE", scale_stride),
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
