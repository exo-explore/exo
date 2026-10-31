"""Fused Metal kernels for Qwen3.8-Flash-Next (qwen4_exp).

At decode, a step of this model is thousands of tiny dependent GPU operations: each of the
48 layers mixes four residual streams twice (hyper-connections), routes 10 of 512 experts
and runs a Gated DeltaNet or attention block, mostly as chains of elementwise ops, casts
and small reductions. Each dependent op costs a few microseconds whatever its size, so
these kernels fuse the chains into one launch each.

Each kernel computes exactly what the MLX op chain it replaces computes, bit for bit:
- float32 where the model computes in float32 (norm statistics and gains, the router's
  softmax and top-k, the Gated DeltaNet decay);
- bfloat16 elsewhere, rounded where the MLX ops round: MLX evaluates bfloat16 math in
  bfloat16 arithmetic, and the elementwise helpers below are MLX's own operator
  definitions;
- reductions in MLX's own order: rms_norm's split of a row over threads, qmv's split of
  a matrix row over the lanes of a simdgroup, and the column reduction's partial sums.
The tests (tests/test_qwen4_exp_kernels.py) check each against the MLX ops.

Callers use them only on Metal (``available()``).
"""

from collections.abc import Callable, Sequence
from typing import cast

import mlx.core as mx

# The most tokens one call of these kernels takes. Their matrix-vector products
# reproduce MLX's qmv, which MLX runs for a single row; the model calls them for
# one-token steps (qwen4_exp._FUSED_TOKENS).
MAX_VECTOR_TOKENS = 8

_HEADER = """
// MLX's own definitions (unary_ops.h, binary_ops.h), so bfloat16 inputs round where the
// MLX ops round.
template <typename T>
inline T mlx_sigmoid(T x) {
  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}

template <typename T>
inline T mlx_logaddexp(T x, T y) {
  if (metal::isnan(x) || metal::isnan(y)) {
    return metal::numeric_limits<T>::quiet_NaN();
  }
  constexpr T inf = metal::numeric_limits<T>::infinity();
  T maxval = metal::max(x, y);
  T minval = metal::min(x, y);
  return (minval == -inf || maxval == inf)
      ? maxval
      : (maxval + log1p(metal::exp(minval - maxval)));
}
"""


def available() -> bool:
    return mx.metal.is_available()


def _norm_per_thread(dims: int) -> int:
    """Elements per thread of the norm kernels: few, with at most 1024 threads."""
    # MLX's rms_norm reads 4 elements per thread
    for per in (4, 8, 16, 32):
        threads = dims // per
        if dims % per == 0 and threads <= 1024 and threads % 32 == 0:
            return per
    raise ValueError(f"no thread split for {dims}")


def norm_supported(dims: int) -> bool:
    """The hyper-connection kernels split a stream of ``dims`` over whole simdgroups."""
    try:
        _norm_per_thread(dims)
    except ValueError:
        return False
    return True


def router_supported(num_experts: int) -> bool:
    """The router kernel takes one thread per expert in one threadgroup, and splits them
    as MLX's softmax splits a row (4 per thread, in simdgroups)."""
    return num_experts % 32 == 0 and num_experts <= 1024


_kernels: dict[str, object] = {}


def _kernel(
    name: str, inputs: Sequence[str], outputs: Sequence[str], source: str
) -> object:
    kernel = _kernels.get(name)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=name,
            input_names=list(inputs),
            output_names=list(outputs),
            source=source,
            header=_HEADER,
        )
        _kernels[name] = kernel
    return kernel


def _call(
    kernel: object,
    inputs: Sequence[mx.array | float | int],
    template: Sequence[tuple[str, mx.Dtype | int | bool]],
    grid: tuple[int, int, int],
    threadgroup: tuple[int, int, int],
    output_shapes: Sequence[Sequence[int]],
    output_dtypes: Sequence[mx.Dtype],
) -> list[mx.array]:
    fn = cast(Callable[..., list[mx.array]], kernel)
    return fn(
        inputs=list(inputs),
        template=list(template),
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[tuple(shape) for shape in output_shapes],
        output_dtypes=list(output_dtypes),
    )


# ------------------------------------------------------------------ hyper-connections

_HC_HEAD = """
    // One threadgroup per (stream, token); thread t owns elements [t * PER, t * PER +
    // PER), as MLX's rms_norm kernel splits a row (so the sums of squares round alike).
    // All loads are issued before any store.
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint s = threadgroup_position_in_grid.x;
    uint m = threadgroup_position_in_grid.y;
    constexpr int TPB = D / PER;
    size_t base = size_t(m) * (S * D) + s * D + tid * PER;
    float vals[PER];
    float ss = 0.0f;
"""

_HC_ADD = """
    // inject = 2 * sigmoid(raw / hc_count); h += block_out * inject
    T hv[PER];
    T ov[PER];
    for (int i = 0; i < PER; i++) {
        hv[i] = h[base + i];
        ov[i] = out[size_t(m) * D + tid * PER + i];
    }
    T raw = inj[size_t(m) * INJ_LD + INJ_OFF + s];
    T q = raw / T(HC);
    T inj_t = T(2) * mlx_sigmoid(q);
    for (int i = 0; i < PER; i++) {
        T prod = ov[i] * inj_t;
        T x = hv[i] + prod;
        h_out[base + i] = x;
        float xf = float(x);
        vals[i] = xf;
        ss += xf * xf;
    }
"""

_HC_LOAD = """
    for (int i = 0; i < PER; i++) {
        vals[i] = float(h[base + i]);
    }
    for (int i = 0; i < PER; i++) {
        ss += vals[i] * vals[i];
    }
"""

_HC_NORM = """
    // ZeroCenteredRMSNorm: rms_norm in float32 (reduced as MLX's kernel reduces),
    // times (1 + w), rounded once.
    float wv[PER];
    for (int i = 0; i < PER; i++) {
        wv[i] = float(w[s * D + tid * PER + i]);
    }
    threadgroup float partial[32];
    threadgroup float inv_mean[1];
    ss = simd_sum(ss);
    if (sg == 0) {
        partial[lane] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        partial[sg] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float total = simd_sum(partial[lane]);
        if (lane == 0) {
            inv_mean[0] = metal::precise::rsqrt(total / float(D) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = inv_mean[0];
    for (int i = 0; i < PER; i++) {
        normed[base + i] = T(vals[i] * inv * (1.0f + wv[i]));
    }
"""


def hc_norm(h: mx.array, weight: mx.array, eps: float, streams: int) -> mx.array:
    """``ZeroCenteredRMSNorm(group_size=D)`` of the streams ``h`` (..., S * D)."""
    width = h.shape[-1]
    dims = width // streams
    tokens = h.size // width
    per = _norm_per_thread(dims)
    threads = dims // per
    kernel = _kernel(
        "qwen4_exp_hc_norm",
        ["h", "w", "eps"],
        ["normed"],
        _HC_HEAD + _HC_LOAD + _HC_NORM,
    )
    (normed,) = _call(
        kernel,
        [h, weight, eps],
        [("T", h.dtype), ("D", dims), ("S", streams), ("PER", per)],
        grid=(threads * streams, tokens, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[h.shape],
        output_dtypes=[h.dtype],
    )
    return normed


def hc_add_norm(
    h: mx.array,
    block_out: mx.array,
    projection: mx.array,
    inject_offset: int,
    hc_count: int,
    weight: mx.array | None,
    eps: float,
) -> tuple[mx.array, mx.array | None]:
    """``add_to_streams(h, block_out, 2 * sigmoid(raw / hc_count))``, with ``raw`` the
    columns ``inject_offset:inject_offset + S`` of ``projection``, and (with ``weight``)
    the zero-centred norm of the result."""
    width = h.shape[-1]
    streams = hc_count
    dims = width // streams
    tokens = h.size // width
    per = _norm_per_thread(dims)
    threads = dims // per
    write_norm = weight is not None
    name = "qwen4_exp_hc_add_norm" if write_norm else "qwen4_exp_hc_add"
    source = _HC_HEAD + _HC_ADD + (_HC_NORM if write_norm else "")
    inputs: list[mx.array | float | int] = [h, block_out, projection]
    input_names = ["h", "out", "inj"]
    if write_norm:
        assert weight is not None
        inputs += [weight, eps]
        input_names += ["w", "eps"]
    outputs = ["h_out", "normed"] if write_norm else ["h_out"]
    kernel = _kernel(name, input_names, outputs, source)
    results = _call(
        kernel,
        inputs,
        [
            ("T", h.dtype),
            ("D", dims),
            ("S", streams),
            ("PER", per),
            ("HC", hc_count),
            ("INJ_LD", projection.shape[-1]),
            ("INJ_OFF", inject_offset),
        ],
        grid=(threads * streams, tokens, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[h.shape] * len(outputs),
        output_dtypes=[h.dtype] * len(outputs),
    )
    return results[0], (results[1] if write_norm else None)


_MIX_UP_SOURCE = """
    // One simdgroup per output column j. The up projection rows j + s * D of the four
    // streams are each reduced by the whole simdgroup exactly as MLX's qmv kernel
    // reduces a row (same lane split, same float expression, simd_sum), then
    // sigmoid, times the normed streams, and the mean over the streams (a bfloat16 sum
    // in stream order, times 1/4, as MLX's mean over that axis).
    constexpr int PACK = 8;              // 4-bit values per uint32
    constexpr int BLOCK = PACK * 32;     // values per simdgroup pass
    constexpr int KW = K / PACK;         // uint32 words per row
    constexpr int NG = K / GS;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sgid = simdgroup_index_in_threadgroup;
    uint j = threadgroup_position_in_grid.x * SGS + sgid;
    threadgroup T act[MT * K];
    for (int i = tid; i < MT * K; i += 32 * SGS) {
        int mm = i / K;
        int kk = i - mm * K;
        // weights = silu(down / hc_count), as MLX's compiled silu: x * sigmoid(x)
        T t = a[size_t(mm) * LDA + kk] / T(HC);
        act[i] = t * mlx_sigmoid(t);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (j >= D) {
        return;
    }
    // Load the four rows' words, scales and biases (one pass of BLOCK values and the
    // remainder) before any arithmetic.
    constexpr int NB = (K - 1) / BLOCK;          // full passes: k < K - BLOCK
    constexpr int TAIL = K - NB * BLOCK;         // values left for the remainder
    uint16_t wl[4][NB + 1][PACK / 4];
    float scl[4][NB + 1];
    float bil[4][NB + 1];
    int rem = metal::clamp(int(TAIL - int(lane) * PACK), 0, PACK);
    for (int s = 0; s < 4; s++) {
        size_t row = size_t(s) * D + j;
        const device uint16_t* wrow = (const device uint16_t*)(wq + row * KW);
        for (int b = 0; b <= NB; b++) {
            int base = b * BLOCK + lane * PACK;
            bool live = b < NB || rem > 0;
            for (int i = 0; i < PACK / 4; i++) {
                wl[s][b][i] = live ? wrow[base / 4 + i] : uint16_t(0);
            }
            scl[s][b] = live ? float(sc[row * NG + base / GS]) : 0.0f;
            bil[s][b] = live ? float(bi[row * NG + base / GS]) : 0.0f;
        }
    }
    for (int mm = 0; mm < MT; mm++) {
        const threadgroup T* x = act + mm * K;
        float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int b = 0; b <= NB; b++) {
            int n = b < NB ? PACK : rem;
            if (n <= 0) {
                continue;
            }
            int base = b * BLOCK + lane * PACK;
            float xt[PACK];
            float sum = 0.0f;
            for (int i = 0; i < n; i += 4) {
                sum += float(x[base + i]) + float(x[base + i + 1]) + float(x[base + i + 2]) + float(x[base + i + 3]);
                xt[i] = x[base + i];
                xt[i + 1] = x[base + i + 1] / 16.0f;
                xt[i + 2] = x[base + i + 2] / 256.0f;
                xt[i + 3] = x[base + i + 3] / 4096.0f;
            }
            for (int s = 0; s < 4; s++) {
                float accum = 0.0f;
                for (int i = 0; i < n / 4; i++) {
                    uint16_t wv = wl[s][b][i];
                    accum += (xt[4 * i] * (wv & 0x000f) + xt[4 * i + 1] * (wv & 0x00f0) +
                              xt[4 * i + 2] * (wv & 0x0f00) + xt[4 * i + 3] * (wv & 0xf000));
                }
                result[s] += scl[s][b] * accum + sum * bil[s][b];
            }
        }
        T total = T(0);
        for (int s = 0; s < 4; s++) {
            float r = simd_sum(result[s]);
            T w = mlx_sigmoid(T(r));
            T p = w * normed[size_t(mm) * 4 * D + s * D + j];
            total = s == 0 ? p : T(total + p);
        }
        if (lane == 0) {
            mixed[size_t(mm) * D + j] = total * T(0.25f);
        }
    }
"""

_MIX_SGS = 8


def hc_mix_up(
    projection: mx.array,
    up_weight: mx.array,
    up_scales: mx.array,
    up_biases: mx.array,
    group_size: int,
    bits: int,
    normed: mx.array,
    hc_count: int,
    low_rank: int,
) -> mx.array:
    """``mean_s(sigmoid(up(silu(projection[..., :low_rank] / hc_count))) * normed)``
    for at most MAX_VECTOR_TOKENS tokens, with ``up`` the affine-quantised
    ``input_mix_weight_up``. Four streams."""
    assert hc_count == 4 and bits == 4 and low_rank % 8 == 0
    width = normed.shape[-1]
    dims = width // hc_count
    tokens = normed.size // width
    assert tokens <= MAX_VECTOR_TOKENS
    kernel = _kernel(
        "qwen4_exp_hc_mix_up",
        ["a", "wq", "sc", "bi", "normed"],
        ["mixed"],
        _MIX_UP_SOURCE,
    )
    (mixed,) = _call(
        kernel,
        [projection, up_weight, up_scales, up_biases, normed],
        [
            ("T", normed.dtype),
            ("D", dims),
            ("K", low_rank),
            ("GS", group_size),
            ("LDA", projection.shape[-1]),
            ("HC", hc_count),
            ("MT", tokens),
            ("SGS", _MIX_SGS),
        ],
        grid=(32 * _MIX_SGS * ((dims + _MIX_SGS - 1) // _MIX_SGS), 1, 1),
        threadgroup=(32 * _MIX_SGS, 1, 1),
        output_shapes=[(*normed.shape[:-1], dims)],
        output_dtypes=[normed.dtype],
    )
    return mixed


# -------------------------------------------------------------------------- attention

_HEAD_NORM_SOURCE = """
    // ZeroCenteredRMSNorm over each head (one threadgroup per (head, token)) of a strided
    // input, with one gain for every head; the output is contiguous (tokens, heads, D).
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint s = threadgroup_position_in_grid.x;
    uint m = threadgroup_position_in_grid.y;
    size_t in_base = size_t(m) * LDX + size_t(s) * SX + tid * PER;
    size_t base = (size_t(m) * S + s) * D + tid * PER;
    float vals[PER];
    float ss = 0.0f;
    for (int i = 0; i < PER; i++) {
        vals[i] = float(x[in_base + i]);
    }
    for (int i = 0; i < PER; i++) {
        ss += vals[i] * vals[i];
    }
""" + _HC_NORM.replace("w[s * D + tid * PER + i]", "w[tid * PER + i]")


def head_norm(
    x: mx.array, weight: mx.array, eps: float, heads: int, head_stride: int
) -> mx.array:
    """``ZeroCenteredRMSNorm(D)`` of ``heads`` heads of D = weight.size values per token,
    head h starting at column h * head_stride of x (..., width). Returns (..., heads, D)."""
    dims = weight.shape[0]
    width = x.shape[-1]
    tokens = x.size // width
    per = _norm_per_thread(dims)
    kernel = _kernel(
        "qwen4_exp_head_norm", ["x", "w", "eps"], ["normed"], _HEAD_NORM_SOURCE
    )
    (normed,) = _call(
        kernel,
        [x, weight, eps],
        [
            ("T", x.dtype),
            ("D", dims),
            ("S", heads),
            ("PER", per),
            ("LDX", width),
            ("SX", head_stride),
        ],
        grid=((dims // per) * heads, tokens, 1),
        threadgroup=(dims // per, 1, 1),
        output_shapes=[(*x.shape[:-1], heads, dims)],
        output_dtypes=[x.dtype],
    )
    return normed


_GATE_SOURCE = """
    // out (B, H, L, D) from attention, times sigmoid of the head's gate (columns
    // [h * 2D + D, h * 2D + 2D) of the query projection), as (B, L, H * D).
    uint d = thread_position_in_grid.x;
    uint h = thread_position_in_grid.y;
    uint bl = thread_position_in_grid.z;
    uint b = bl / L;
    uint l = bl - b * L;
    T o = attn[((size_t(b) * H + h) * L + l) * D + d];
    T g = q[size_t(bl) * (H * 2 * D) + h * 2 * D + D + d];
    gated[(size_t(bl) * H + h) * D + d] = o * mlx_sigmoid(g);
"""


def gate_heads(attention: mx.array, query_projection: mx.array) -> mx.array:
    """``out.transpose(0, 2, 1, 3).reshape(B, L, -1) * sigmoid(gate)`` with the gate
    interleaved after each head's query in ``query_projection`` (B, L, H * 2D)."""
    batch, heads, length, dims = attention.shape
    kernel = _kernel("qwen4_exp_gate_heads", ["attn", "q"], ["gated"], _GATE_SOURCE)
    (gated,) = _call(
        kernel,
        [attention, query_projection],
        [("T", attention.dtype), ("H", heads), ("L", length), ("D", dims)],
        grid=(dims, heads, batch * length),
        threadgroup=(min(256, dims), 1, 1),
        output_shapes=[(batch, length, heads * dims)],
        output_dtypes=[attention.dtype],
    )
    return gated


# -------------------------------------------------------------------------------- MoE

_ROUTER_SOURCE = """
    // One threadgroup of E threads for each token. Softmax over the E router logits in
    // float32 (as MLX's softmax), the top K (each simdgroup picks its own top K, then
    // the first simdgroup merges them), their renormalised scores, and the shared
    // expert's sigmoid gate (logit column E) as an extra (K+1)-th expert E.
    constexpr int NSG = E / 32;
    constexpr int CPL = (NSG * K + 31) / 32;   // merge candidates per lane
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint m = threadgroup_position_in_grid.y;
    threadgroup float red[NSG];
    threadgroup float part4[E / 4];
    threadgroup float local[32];
    threadgroup float cand_v[NSG * K];
    threadgroup int cand_i[NSG * K];
    float v = float(logits[size_t(m) * LD + tid]);
    float mx_ = simd_max(v);
    if (lane == 0) {
        red[sg] = mx_;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx_ = red[0];
    for (int i = 1; i < NSG; i++) {
        mx_ = metal::max(mx_, red[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float e = metal::fast::exp(v - mx_);
    // The normaliser, summed as MLX's softmax sums it: each of its threads adds 4
    // consecutive values in order, simd_sum over 32 such threads, then simd_sum over
    // those partials (zero-padded)
    float e1 = simd_shuffle_down(e, 1);
    float e2 = simd_shuffle_down(e, 2);
    float e3 = simd_shuffle_down(e, 3);
    if (tid % 4 == 0) {
        part4[tid / 4] = ((e + e1) + e2) + e3;
    }
    if (tid < 32) {
        local[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr int VT = E / 4;              // MLX softmax threads for the row
    constexpr int VSG = (VT + 31) / 32;    // and their simdgroups
    if (int(sg) < VSG) {
        int at = sg * 32 + lane;
        float partial = simd_sum(at < VT ? part4[at] : 0.0f);
        if (lane == 0) {
            local[sg] = partial;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float s = simd_sum(local[lane]);
        if (lane == 0) {
            red[0] = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = red[0];
    float p = e * (1 / total);
    // This simdgroup's top K, largest first (ties: lower expert first)
    float cand = p;
    for (int k = 0; k < K; k++) {
        float bv = cand;
        int bidx = tid;
        for (int off = 16; off > 0; off /= 2) {
            float ov = simd_shuffle_xor(bv, off);
            int oi = simd_shuffle_xor(bidx, off);
            if (ov > bv || (ov == bv && oi < bidx)) {
                bv = ov;
                bidx = oi;
            }
        }
        if (lane == 0) {
            cand_v[sg * K + k] = bv;
            cand_i[sg * K + k] = bidx;
        }
        if (int(tid) == bidx) {
            cand = -1.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0) {
        return;
    }
    float lv[CPL];
    int li[CPL];
    for (int c = 0; c < CPL; c++) {
        int at = lane + 32 * c;
        lv[c] = at < NSG * K ? cand_v[at] : -2.0f;
        li[c] = at < NSG * K ? cand_i[at] : E;
    }
    float sel_v[K];
    int sel_i[K];
    for (int k = 0; k < K; k++) {
        float bv = lv[0];
        int bidx = li[0];
        for (int c = 1; c < CPL; c++) {
            if (lv[c] > bv || (lv[c] == bv && li[c] < bidx)) {
                bv = lv[c];
                bidx = li[c];
            }
        }
        for (int off = 16; off > 0; off /= 2) {
            float ov = simd_shuffle_xor(bv, off);
            int oi = simd_shuffle_xor(bidx, off);
            if (ov > bv || (ov == bv && oi < bidx)) {
                bv = ov;
                bidx = oi;
            }
        }
        sel_v[k] = bv;
        sel_i[k] = bidx;
        for (int c = 0; c < CPL; c++) {
            if (li[c] == bidx) {
                lv[c] = -2.0f;
            }
        }
    }
    if (lane == 0) {
        float ssum = 0.0f;
        for (int k = 0; k < K; k++) {
            ssum += sel_v[k];
        }
        for (int k = 0; k < K; k++) {
            float score = NORM ? sel_v[k] / ssum : sel_v[k];
            idx[size_t(m) * (K + 1) + k] = uint32_t(sel_i[k]);
            both[size_t(m) * 2 * (K + 1) + k] = uint32_t(sel_i[k]);
            both[size_t(m) * 2 * (K + 1) + K + 1 + k] = uint32_t(sel_i[k] + E + 1);
            wts[size_t(m) * (K + 1) + k] = T(score);
        }
        idx[size_t(m) * (K + 1) + K] = uint32_t(E);
        both[size_t(m) * 2 * (K + 1) + K] = uint32_t(E);
        both[size_t(m) * 2 * (K + 1) + 2 * K + 1] = uint32_t(2 * E + 1);
        wts[size_t(m) * (K + 1) + K] = mlx_sigmoid(logits[size_t(m) * LD + E]);
    }
"""


def moe_route(
    logits: mx.array, num_experts: int, top_k: int, normalise: bool
) -> tuple[mx.array, mx.array, mx.array]:
    """From (..., E + 1) logits (the router's E, then the shared expert gate's): the
    experts (..., top_k + 1) and their weights, with the shared expert last as expert E
    weighted by the sigmoid of its gate; and the same experts' rows in an array of the
    E + 1 experts' gate projections followed by their up projections (..., 2 top_k + 2)."""
    assert router_supported(num_experts)
    tokens = logits.size // logits.shape[-1]
    kernel = _kernel(
        "qwen4_exp_moe_route", ["logits"], ["idx", "wts", "both"], _ROUTER_SOURCE
    )
    shape = (*logits.shape[:-1], top_k + 1)
    idx, wts, both = _call(
        kernel,
        [logits],
        [
            ("T", logits.dtype),
            ("E", num_experts),
            ("K", top_k),
            ("LD", logits.shape[-1]),
            ("NORM", normalise),
        ],
        grid=(num_experts, tokens, 1),
        threadgroup=(num_experts, 1, 1),
        output_shapes=[shape, shape, (*logits.shape[:-1], 2 * (top_k + 1))],
        output_dtypes=[mx.uint32, logits.dtype, mx.uint32],
    )
    return idx, wts, both


_COMBINE_SOURCE = """
    // sum_k y_k * w_k over the routed experts, accumulated in bfloat16 in the order
    // MLX's reduction over that axis uses (8 interleaved partial sums, then in turn),
    // plus the shared expert's y * w.
    uint j = thread_position_in_grid.x;
    uint m = thread_position_in_grid.y;
    if (j >= D) {
        return;
    }
    constexpr int NR = KR < 8 ? KR : 8;
    T partial[NR];
    for (int r = 0; r < NR; r++) {
        partial[r] = y[(size_t(m) * (KR + 1) + r) * D + j] * wts[size_t(m) * (KR + 1) + r];
        for (int e = r + 8; e < KR; e += 8) {
            T p = y[(size_t(m) * (KR + 1) + e) * D + j] * wts[size_t(m) * (KR + 1) + e];
            partial[r] = partial[r] + p;
        }
    }
    T routed = partial[0];
    for (int r = 1; r < NR; r++) {
        routed = routed + partial[r];
    }
    T shared = y[(size_t(m) * (KR + 1) + KR) * D + j] * wts[size_t(m) * (KR + 1) + KR];
    out[size_t(m) * D + j] = routed + shared;
"""


def moe_combine(expert_out: mx.array, weights: mx.array) -> mx.array:
    """(..., K + 1, D) expert outputs, (..., K + 1) weights -> (..., D)."""
    dims = expert_out.shape[-1]
    experts = expert_out.shape[-2]
    tokens = expert_out.size // (dims * experts)
    kernel = _kernel("qwen4_exp_moe_combine", ["y", "wts"], ["out"], _COMBINE_SOURCE)
    (out,) = _call(
        kernel,
        [expert_out, weights],
        [("T", expert_out.dtype), ("D", dims), ("KR", experts - 1)],
        grid=(dims, tokens, 1),
        threadgroup=(min(256, dims), 1, 1),
        output_shapes=[(*expert_out.shape[:-2], dims)],
        output_dtypes=[expert_out.dtype],
    )
    return out


_DOWN_SOURCE = """
    // One expert's down projection for one token (threadgroup (x, expert, token)):
    // act = silu(gate) * up (as mlx-lm's compiled swiglu) in threadgroup memory, then RPS
    // rows per simdgroup, each reduced by the whole simdgroup exactly as MLX's qmv kernel
    // reduces a row (8 values per lane per 256-value pass, the remainder last, float
    // load_vector / qdot expressions, simd_sum).
    constexpr int PACK = 8;
    constexpr int BLOCK = PACK * 32;
    constexpr int KW = K / PACK;
    constexpr int NG = K / GS;
    constexpr int NB = (K - 1) / BLOCK;
    constexpr int TAIL = K - NB * BLOCK;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sgid = simdgroup_index_in_threadgroup;
    uint e = threadgroup_position_in_grid.y;
    uint m = threadgroup_position_in_grid.z;
    threadgroup T act[K];
    // gu: (tokens, 2 KE, K), the KE experts' gate projections, then their up projections
    const device T* gate_in = gu + (size_t(m) * 2 * KE + e) * K;
    const device T* up_in = gu + (size_t(m) * 2 * KE + KE + e) * K;
    for (int i = tid; i < K; i += 32 * SGS) {
        T g = gate_in[i];
        T u = up_in[i];
        T sg = g * mlx_sigmoid(g);
        act[i] = sg * u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    size_t expert_rows = size_t(idx[size_t(m) * KE + e]) * D;
    int j0 = (threadgroup_position_in_grid.x * SGS + sgid) * RPS;
    int rem = metal::clamp(int(TAIL - int(lane) * PACK), 0, PACK);
    float result[RPS];
    for (int r = 0; r < RPS; r++) {
        result[r] = 0.0f;
    }
    for (int b = 0; b <= NB; b++) {
        int n = b < NB ? PACK : rem;
        if (n <= 0) {
            continue;
        }
        int base = b * BLOCK + lane * PACK;
        float xt[PACK];
        float sum = 0.0f;
        for (int i = 0; i < n; i += 4) {
            sum += float(act[base + i]) + float(act[base + i + 1]) + float(act[base + i + 2]) + float(act[base + i + 3]);
            xt[i] = act[base + i];
            xt[i + 1] = act[base + i + 1] / 16.0f;
            xt[i + 2] = act[base + i + 2] / 256.0f;
            xt[i + 3] = act[base + i + 3] / 4096.0f;
        }
        for (int r = 0; r < RPS; r++) {
            int j = j0 + r;
            if (j >= D) {
                break;
            }
            size_t row = expert_rows + j;
            uint32_t word = wq[row * KW + base / PACK];
            float scale = sc[row * NG + base / GS];
            float bias = bi[row * NG + base / GS];
            uint16_t ws[2] = {uint16_t(word & 0xffffu), uint16_t(word >> 16)};
            float accum = 0.0f;
            for (int i = 0; i < n / 4; i++) {
                uint16_t wv = ws[i];
                accum += (xt[4 * i] * (wv & 0x000f) + xt[4 * i + 1] * (wv & 0x00f0) +
                          xt[4 * i + 2] * (wv & 0x0f00) + xt[4 * i + 3] * (wv & 0xf000));
            }
            result[r] += scale * accum + sum * bias;
        }
    }
    for (int r = 0; r < RPS; r++) {
        int j = j0 + r;
        float total = simd_sum(result[r]);
        if (lane == 0 && j < D) {
            y[(size_t(m) * KE + e) * D + j] = T(total);
        }
    }
"""

_DOWN_SGS = 8
_DOWN_RPS = 4


def moe_down(
    gate_up: mx.array,
    indices: mx.array,
    down_weight: mx.array,
    down_scales: mx.array,
    down_biases: mx.array,
    group_size: int,
) -> mx.array:
    """For (..., 2 KE, K) gate then up projections of each token's KE experts and their
    (..., KE) indices: each expert's 4-bit down projection (E, D, K) of
    swiglu(gate, up), (..., KE, D), as mlx-lm's SwitchGLU computes it."""
    experts = indices.shape[-1]
    inner = gate_up.shape[-1]
    dims = down_weight.shape[1]
    tokens = indices.size // experts
    assert inner % 8 == 0 and inner * 2 <= 32768
    kernel = _kernel(
        "qwen4_exp_moe_down_v2", ["gu", "idx", "wq", "sc", "bi"], ["y"], _DOWN_SOURCE
    )
    per_group = _DOWN_SGS * _DOWN_RPS
    (y,) = _call(
        kernel,
        [gate_up, indices, down_weight, down_scales, down_biases],
        [
            ("T", gate_up.dtype),
            ("D", dims),
            ("K", inner),
            ("GS", group_size),
            ("KE", experts),
            ("SGS", _DOWN_SGS),
            ("RPS", _DOWN_RPS),
        ],
        grid=(32 * _DOWN_SGS * ((dims + per_group - 1) // per_group), experts, tokens),
        threadgroup=(32 * _DOWN_SGS, 1, 1),
        output_shapes=[(*indices.shape, dims)],
        output_dtypes=[gate_up.dtype],
    )
    return y


# --------------------------------------------------------------------- Gated DeltaNet

_GDN_PREPARE_SOURCE = """
    // Threadgroup y: a q head, a k head, a v head, or (the last) the decay and beta.
    // Depthwise causal conv over [conv state, tokens], SiLU, and the L2 norm of q and k
    // with the head_dim ** -0.5 readout scale folded into q (mlx-lm's normalize_qk).
    constexpr int CD = 2 * HK * DK + HV * DV;
    constexpr int KS = KC - 1;
    constexpr int PER = DK / 32;
    uint lane = thread_position_in_threadgroup.x;
    uint slot = threadgroup_position_in_grid.y;
    if (slot == 2 * HK + HV) {
        for (int mm = 0; mm < MT; mm++) {
            for (int hv = lane; hv < HV; hv += 32) {
                T b = proj[size_t(mm) * NP + CD + HV * DV + hv];
                T a = proj[size_t(mm) * NP + CD + HV * DV + HV + hv];
                beta[mm * HV + hv] = mlx_sigmoid(b);
                // compute_g: exp(-exp(A_log) * softplus(a + dt_bias))
                T sp = mlx_logaddexp(T(a + dt_bias[hv]), T(0));
                g[mm * HV + hv] = metal::precise::exp(-metal::precise::exp(float(a_log[hv])) * float(sp));
            }
        }
        return;
    }
    int base;
    int kind;  // 0 q, 1 k, 2 v
    device T* dst;
    int heads;
    int head;
    if (int(slot) < HK) {
        kind = 0; head = slot; heads = HK; base = head * DK; dst = q_out;
    } else if (int(slot) < 2 * HK) {
        kind = 1; head = slot - HK; heads = HK; base = HK * DK + head * DK; dst = k_out;
    } else {
        kind = 2; head = slot - 2 * HK; heads = HV; base = 2 * HK * DK + head * DV; dst = v_out;
    }
    // Lane l owns channels [l * PER, l * PER + PER) of the head, as MLX's rms_norm
    // splits a 128-wide row.
    T vals[MT][PER];
    for (int p = 0; p < PER; p++) {
        int c = base + lane * PER + p;
        float w[KC];
        for (int kk = 0; kk < KC; kk++) {
            w[kk] = float(cw[c * KC + kk]);
        }
        for (int mm = 0; mm < MT; mm++) {
            float acc = 0.0f;
            for (int kk = 0; kk < KC; kk++) {
                int src = mm - KS + kk;
                T xv = src < 0 ? state[(KS + src) * CD + c] : proj[size_t(src) * NP + c];
                acc += w[kk] * float(xv);
            }
            T co = T(acc);
            vals[mm][p] = co * mlx_sigmoid(co);
        }
        for (int t = 0; t < KS; t++) {
            int src = MT - KS + t;
            new_state[t * CD + c] = src < 0 ? state[(KS + src) * CD + c] : proj[size_t(src) * NP + c];
        }
    }
    for (int mm = 0; mm < MT; mm++) {
        if (kind == 2) {
            for (int p = 0; p < PER; p++) {
                dst[(size_t(mm) * heads + head) * DK + lane * PER + p] = vals[mm][p];
            }
            continue;
        }
        float ss = 0.0f;
        for (int p = 0; p < PER; p++) {
            float x = float(vals[mm][p]);
            ss += x * x;
        }
        ss = simd_sum(ss);
        float inv = metal::precise::rsqrt(ss / float(DK) + eps);
        T scale = kind == 0 ? T(q_scale) : T(k_scale);
        for (int p = 0; p < PER; p++) {
            T n = T(float(vals[mm][p]) * inv);
            dst[(size_t(mm) * heads + head) * DK + lane * PER + p] = scale * n;
        }
    }
"""


def gdn_prepare(
    projection: mx.array,
    conv_state: mx.array,
    conv_weight: mx.array,
    a_log: mx.array,
    dt_bias: mx.array,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """For (1, T, NP) projections [qkv | z | b | a] and the (1, kernel - 1, conv_dim)
    conv state: q, k (1, T, Hk, Dk), v (1, T, Hv, Dv), g (1, T, Hv) float32,
    beta (1, T, Hv) and the next conv state, as Qwen3.5's Gated DeltaNet computes them."""
    assert head_k_dim == head_v_dim and head_k_dim % 32 == 0
    batch, tokens, width = projection.shape
    assert batch == 1 and tokens <= MAX_VECTOR_TOKENS
    kernel_size = conv_weight.shape[1]
    conv_dim = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
    inv_scale = cast(float, head_k_dim**-0.5)
    kernel = _kernel(
        "qwen4_exp_gdn_prepare",
        ["proj", "state", "cw", "a_log", "dt_bias", "eps", "q_scale", "k_scale"],
        ["q_out", "k_out", "v_out", "g", "beta", "new_state"],
        _GDN_PREPARE_SOURCE,
    )
    q, k, v, g, beta, new_state = _call(
        kernel,
        [
            projection,
            conv_state,
            conv_weight,
            a_log,
            dt_bias,
            1e-6 * inv_scale**2,
            inv_scale**2,
            inv_scale,
        ],
        [
            ("T", projection.dtype),
            ("HK", num_k_heads),
            ("HV", num_v_heads),
            ("DK", head_k_dim),
            ("DV", head_v_dim),
            ("KC", kernel_size),
            ("MT", tokens),
            ("NP", width),
        ],
        grid=(32, 2 * num_k_heads + num_v_heads + 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[
            (1, tokens, num_k_heads, head_k_dim),
            (1, tokens, num_k_heads, head_k_dim),
            (1, tokens, num_v_heads, head_v_dim),
            (1, tokens, num_v_heads),
            (1, tokens, num_v_heads),
            (1, kernel_size - 1, conv_dim),
        ],
        output_dtypes=[
            projection.dtype,
            projection.dtype,
            projection.dtype,
            mx.float32,
            projection.dtype,
            projection.dtype,
        ],
    )
    return q, k, v, g, beta, new_state


_GATED_NORM_SOURCE = """
    // RMSNorm(out) * weight * sigmoid(z) for one (token, head): one simdgroup.
    constexpr int PER = DV / 32;
    uint lane = thread_position_in_threadgroup.x;
    uint head = threadgroup_position_in_grid.y;
    uint m = threadgroup_position_in_grid.z;
    size_t base = (size_t(m) * HV + head) * DV;
    float vals[PER];
    float ss = 0.0f;
    for (int p = 0; p < PER; p++) {
        float x = float(o[base + lane * PER + p]);
        vals[p] = x;
        ss += x * x;
    }
    ss = simd_sum(ss);
    float inv = metal::precise::rsqrt(ss / float(DV) + eps);
    for (int p = 0; p < PER; p++) {
        int d = lane * PER + p;
        T y = w[d] * T(vals[p] * inv);
        float z = float(proj[size_t(m) * NP + ZOFF + head * DV + d]);
        y_out[base + d] = T(mlx_sigmoid(z) * float(y));
    }
"""


def gdn_gated_norm(
    out: mx.array,
    projection: mx.array,
    z_offset: int,
    weight: mx.array,
    eps: float,
) -> mx.array:
    """The sigmoid-gated RMS norm of the Gated DeltaNet output ``out`` (1, T, Hv, Dv) with
    the gate z read from ``projection`` at ``z_offset``. Returns (1, T, Hv * Dv)."""
    batch, tokens, heads, dims = out.shape
    kernel = _kernel(
        "qwen4_exp_gdn_gated_norm",
        ["o", "proj", "w", "eps"],
        ["y_out"],
        _GATED_NORM_SOURCE,
    )
    (y,) = _call(
        kernel,
        [out, projection, weight, eps],
        [
            ("T", out.dtype),
            ("HV", heads),
            ("DV", dims),
            ("NP", projection.shape[-1]),
            ("ZOFF", z_offset),
        ],
        grid=(32, heads, batch * tokens),
        threadgroup=(32, 1, 1),
        output_shapes=[(batch, tokens, heads * dims)],
        output_dtypes=[out.dtype],
    )
    return y
