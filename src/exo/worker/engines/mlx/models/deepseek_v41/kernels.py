"""Fused Metal kernels for DeepSeek-V4.1's decode path.

Each kernel replaces a chain of small MLX ops with one launch and keeps the arithmetic of
the ops it replaces: float32 where they compute in float32, the same roundings to bf16
between steps. Accumulation orders differ (float32 rounding noise). Every function falls
back to the plain MLX ops when Metal is unavailable or a shape is outside what the kernel
handles, and `set_fused(False)` turns all of them off.
"""

import math
from collections.abc import Callable
from functools import cache

import mlx.core as mx

_ENABLED = [True]
_DISABLED: set[str] = set()


def set_fused(enabled: bool, disabled: frozenset[str] = frozenset()) -> None:
    """Turn the fused kernels on or off (off: the plain MLX ops); `disabled` names single
    kernels to leave off (for measurements)."""
    _ENABLED[0] = enabled
    _DISABLED.clear()
    _DISABLED.update(disabled)


def fused_available(name: str = "") -> bool:
    return (
        _ENABLED[0]
        and name not in _DISABLED
        and mx.metal.is_available()
        and mx.default_device() == mx.gpu
    )


# ---------------------------------------------------------------------------------------
# Kernel source helpers


def _strip(source: str, begin: str, end: str) -> str:
    start = source.index(begin)
    stop = source.index(end) + len(end)
    return source[:start] + source[stop:]


# ---------------------------------------------------------------------------------------
# float32 logits from a bf16 head: the products of bf16 values are exact in float32, so
# this is the float32 matmul of the upcast weights without materializing them.

_HEAD_SOURCE = """
    // each simdgroup computes ROWS output rows for every input row (M <= MAX_M); the
    // weights of a chunk sit in registers, the inputs are re-read (they stay in cache)
    constexpr int VEC = 8;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    int n_rows = w_shape[0];
    int dim = w_shape[1];
    int m = x_shape[0];
    int row0 = (threadgroup_position_in_grid.x * SG + sg) * ROWS;
    float acc[ROWS][MAX_M];
    for (int r = 0; r < ROWS; r++) {
        for (int i = 0; i < MAX_M; i++) {
            acc[r][i] = 0.0f;
        }
    }
    for (int k = lane * VEC; k < dim; k += 32 * VEC) {
        float wv[ROWS][VEC];
        for (int r = 0; r < ROWS; r++) {
            int row = min(row0 + r, n_rows - 1);
            const device bfloat* wr = w + size_t(row) * dim + k;
            for (int v = 0; v < VEC; v++) {
                wv[r][v] = float(wr[v]);
            }
        }
        for (int i = 0; i < MAX_M; i++) {
            if (i < m) {
                const device T* xr = x + size_t(i) * dim + k;
                float xv[VEC];
                for (int v = 0; v < VEC; v++) {
                    xv[v] = float(xr[v]);
                }
                for (int r = 0; r < ROWS; r++) {
                    float s = 0.0f;
                    for (int v = 0; v < VEC; v++) {
                        s += xv[v] * wv[r][v];
                    }
                    acc[r][i] += s;
                }
            }
        }
    }
    for (int r = 0; r < ROWS; r++) {
        int row = row0 + r;
        for (int i = 0; i < MAX_M; i++) {
            float s = simd_sum(acc[r][i]);
            if (lane == 0 && i < m && row < n_rows) {
                out[size_t(i) * n_rows + row] = s;
            }
        }
    }
"""


@cache
def _head_kernel() -> Callable[..., list[mx.array]]:
    return mx.fast.metal_kernel(
        name="dsv41_head_gemv",
        input_names=["x", "w"],
        output_names=["out"],
        source=_HEAD_SOURCE,
    )


def head_logits(x: mx.array, weight: mx.array) -> mx.array:
    """x [..., D] (any float dtype) @ weight [V, D] (bf16) -> float32 [..., V], with
    float32 products and sums."""
    rows = math.prod(x.shape[:-1])
    if (
        fused_available("head")
        and weight.dtype == mx.bfloat16
        and rows <= 8
        and x.shape[-1] % 256 == 0
    ):
        max_m = 1 if rows == 1 else (2 if rows == 2 else (4 if rows <= 4 else 8))
        per_sg_rows = 4 if max_m <= 2 else 2
        sgs = 8
        n_out = weight.shape[0]
        groups = -(-n_out // (per_sg_rows * sgs))
        (out,) = _head_kernel()(
            inputs=[x.reshape(rows, -1), weight],
            template=[
                ("T", x.dtype),
                ("ROWS", per_sg_rows),
                ("MAX_M", max_m),
                ("SG", sgs),
            ],
            grid=(groups * 32 * sgs, 1, 1),
            threadgroup=(32 * sgs, 1, 1),
            output_shapes=[(rows, n_out)],
            output_dtypes=[mx.float32],
        )
        return out.reshape(*x.shape[:-1], n_out)
    return x.astype(mx.float32) @ weight.astype(mx.float32).T


# ---------------------------------------------------------------------------------------
# Hyper-connections. A sublayer boundary is: expand the sublayer's output into the streams,
# compute the next mixes from the new streams (RMS norm, a [24, 4D] float32 matvec, then
# sigmoids, a softmax and Sinkhorn), and collapse the streams with the previous pre mix
# into the next sublayer's normalized input. Two launches: `hc_partials` (many
# threadgroups: the new streams, their sum of squares and the 24 dot products, per chunk)
# and `hc_finish` (one threadgroup per row: the rest).

_HC_CHUNK = 1024  # elements of the flattened streams per threadgroup (256 threads x 4)
_HC_MIX = 24

_HC_PARTIALS_SOURCE = """
    constexpr int E = 4;
    constexpr int NMIX = 24;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    int chunks = CHUNKS;
    int r = threadgroup_position_in_grid.x / chunks;
    int c = threadgroup_position_in_grid.x % chunks;
    int width = fn_shape[1];          // hc * D
    int dim = width / HC;
    int f = c * CHUNK + tid * E;      // flat index into [hc, D]
    int stream = f / dim;
    int d = f % dim;
    const device bfloat* sin_row = streams + size_t(r) * width;
    float v[E];
#if EXPAND
    float post_i = post[r * HC + stream];
    float xin[E];
    for (int e = 0; e < E; e++) {
        xin[e] = float(x[size_t(r) * dim + d + e]);
    }
    for (int e = 0; e < E; e++) {
        float mixed = 0.0f;
        for (int j = 0; j < HC; j++) {
            float cj = comb[(r * HC + j) * HC + stream];
            mixed = mixed + cj * float(sin_row[j * dim + d + e]);
        }
        float o = post_i * xin[e] + mixed;
        bfloat ob = bfloat(o);
        new_streams[size_t(r) * width + f + e] = ob;
        v[e] = float(ob);
    }
#else
    for (int e = 0; e < E; e++) {
        v[e] = float(sin_row[f + e]);
    }
#endif
    float acc[NMIX + 1];
    for (int j = 0; j < NMIX; j++) {
        const device float* w = fn + size_t(j) * width + f;
        float s = 0.0f;
        for (int e = 0; e < E; e++) {
            s += w[e] * v[e];
        }
        acc[j] = s;
    }
    float sq = 0.0f;
    for (int e = 0; e < E; e++) {
        sq += v[e] * v[e];
    }
    acc[NMIX] = sq;
    threadgroup float red[8][NMIX + 1];
    for (int j = 0; j <= NMIX; j++) {
        float s = simd_sum(acc[j]);
        if (lane == 0) {
            red[sg][j] = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid <= NMIX) {
        float s = 0.0f;
        for (int k = 0; k < 8; k++) {
            s += red[k][tid];
        }
        partials[(size_t(r) * chunks + c) * (NMIX + 1) + tid] = s;
    }
"""

_HC_FINISH_SOURCE = """
    constexpr int NMIX = 24;
    constexpr int THREADS = 256;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    int r = threadgroup_position_in_grid.x;
    int chunks = partials_shape[1];
    int dim = norm_w_shape[0];
    int width = HC * dim;
    threadgroup float total[NMIX + 1];
    threadgroup float prev[HC];
    threadgroup float red[8];
    if (tid <= NMIX) {
        float s = 0.0f;
        for (int c = 0; c < chunks; c++) {
            s += partials[(size_t(r) * chunks + c) * (NMIX + 1) + tid];
        }
        total[tid] = s;
    }
    if (tid < HC) {
        prev[tid] = pre_prev[r * HC + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // the mixes: simdgroup 0, lane k < HC * HC owns comb element k
    if (sg == 0) {
        float inv = metal::precise::rsqrt(total[NMIX] / float(width) + params[0]);
        float hc_eps = params[1];
        if (lane < 2 * HC) {
            int j = lane;
            int which = j < HC ? 0 : 1;
            float a = (total[j] * inv) * scale[which] + base[j];
            float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(a)));
            float sig = a < 0 ? y : 1.0f - y;
            if (j < HC) {
                pre_out[r * HC + j] = sig + hc_eps;
            } else {
                post_out[r * HC + j - HC] = 2.0f * sig;
            }
        }
        // comb: softmax over each row of the HC x HC matrix, + eps, then Sinkhorn
        int k = lane < HC * HC ? lane : 0;
        int row_i = k / HC;
        int col_i = k % HC;
        float v = (total[2 * HC + k] * inv) * scale[2] + base[2 * HC + k];
        // row max and sum via shuffles within the row's HC lanes
        float mx_ = v;
        for (int o = 1; o < HC; o++) {
            mx_ = max(mx_, simd_shuffle(v, uint(row_i * HC + (col_i + o) % HC)));
        }
        float e = fast::exp(v - mx_);
        float sum = 0.0f;
        for (int o = 0; o < HC; o++) {
            sum += simd_shuffle(e, uint(row_i * HC + o));
        }
        float m = e * (1.0f / sum) + hc_eps;
        for (int t = 0; t < 2 * ITERS - 1; t++) {
            bool rows = t % 2 == 1;
            // rows: normalize across the row (sum over columns); else across the column
            float s_ = 0.0f;
            for (int o = 0; o < HC; o++) {
                uint src = rows ? uint(row_i * HC + o) : uint(o * HC + col_i);
                s_ += simd_shuffle(m, src);
            }
            m = m / (s_ + hc_eps);
        }
        if (lane < HC * HC) {
            comb_out[r * HC * HC + lane] = m;
        }
    }
    // collapse with the previous pre mix, then the sublayer's RMS norm
    const device bfloat* srow = streams + size_t(r) * width;
    constexpr int PER = MAXPER;
    float cv[PER];
    float sq = 0.0f;
    int n_per = dim / THREADS;
    for (int k = 0; k < PER; k++) {
        if (k < n_per) {
            int d = k * THREADS + tid;
            float s = 0.0f;
            for (int i = 0; i < HC; i++) {
                s = s + prev[i] * float(srow[i * dim + d]);
            }
            float cb = float(bfloat(s));
            cv[k] = cb;
            sq += cb * cb;
        }
    }
    sq = simd_sum(sq);
    if (lane == 0) {
        red[sg] = sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (int k = 0; k < THREADS / 32; k++) {
        tot += red[k];
    }
    float inv_norm = metal::precise::rsqrt(tot / float(dim) + params[2]);
    for (int k = 0; k < PER; k++) {
        if (k < n_per) {
            int d = k * THREADS + tid;
            h_out[size_t(r) * dim + d] = bfloat(float(norm_w[d]) * float(bfloat(cv[k] * inv_norm)));
        }
    }
"""


@cache
def _hc_partials_kernel(expand: bool) -> Callable[..., list[mx.array]]:
    source = (
        "#define EXPAND 1\n" if expand else "#define EXPAND 0\n"
    ) + _HC_PARTIALS_SOURCE
    inputs = ["streams", "fn"] + (["x", "post", "comb"] if expand else [])
    outputs = (["new_streams"] if expand else []) + ["partials"]
    return mx.fast.metal_kernel(
        name=f"dsv41_hc_partials_{'expand' if expand else 'plain'}",
        input_names=inputs,
        output_names=outputs,
        source=source,
    )


@cache
def _hc_finish_kernel() -> Callable[..., list[mx.array]]:
    return mx.fast.metal_kernel(
        name="dsv41_hc_finish",
        input_names=[
            "partials",
            "streams",
            "pre_prev",
            "scale",
            "base",
            "norm_w",
            "params",
        ],
        output_names=["pre_out", "post_out", "comb_out", "h_out"],
        source=_HC_FINISH_SOURCE,
    )


def hc_fusable(streams: mx.array) -> bool:
    """Whether the fused hyper-connection kernels take these streams [B, L, hc, D]: decode
    and short verify steps (the matvec form; prefill keeps the matmul)."""
    if not fused_available("hc") or streams.dtype != mx.bfloat16:
        return False
    rows = streams.shape[0] * streams.shape[1]
    width = streams.shape[2] * streams.shape[3]
    return rows <= 16 and width % _HC_CHUNK == 0 and streams.shape[3] % 256 == 0


def hc_partials(
    streams: mx.array,
    fn: mx.array,
    expand: tuple[mx.array, mx.array, mx.array] | None = None,
) -> tuple[mx.array, mx.array]:
    """Returns (streams, partials): with expand = (x, post, comb), the streams after the
    expand (post * x + comb-mixed streams, rounded to the streams' dtype); then, for the
    next mixes, per chunk the 24 dot products with fn and the sum of squares."""
    batch, length, hc, dim = streams.shape
    rows = batch * length
    width = hc * dim
    chunks = width // _HC_CHUNK
    template = [("HC", hc), ("CHUNK", _HC_CHUNK), ("CHUNKS", chunks)]
    common = dict(
        grid=(rows * chunks * 256, 1, 1),
        threadgroup=(256, 1, 1),
    )
    if expand is None:
        (partials,) = _hc_partials_kernel(False)(
            inputs=[streams, fn],
            template=template,
            output_shapes=[(rows, chunks, _HC_MIX + 1)],
            output_dtypes=[mx.float32],
            **common,
        )
        return streams, partials
    x, post, comb = expand
    new_streams, partials = _hc_partials_kernel(True)(
        inputs=[
            streams,
            fn,
            x.astype(mx.bfloat16),
            post.astype(mx.float32),
            comb.astype(mx.float32),
        ],
        template=template,
        output_shapes=[streams.shape, (rows, chunks, _HC_MIX + 1)],
        output_dtypes=[streams.dtype, mx.float32],
        **common,
    )
    return new_streams, partials


def hc_finish(
    partials: mx.array,
    streams: mx.array,
    pre_prev: mx.array,
    *,
    scale: mx.array,
    base: mx.array,
    norm_weight: mx.array,
    norm_eps: float,
    hc_eps: float,
    iters: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """(pre, post, comb) of the next mixes, and the streams collapsed with pre_prev and
    RMS-normalized with norm_weight (the next sublayer's input)."""
    batch, length, hc, dim = streams.shape
    rows = batch * length
    params = mx.array([norm_eps, hc_eps, norm_eps], dtype=mx.float32)
    pre, post, comb, h = _hc_finish_kernel()(
        inputs=[
            partials,
            streams,
            pre_prev.astype(mx.float32),
            scale.astype(mx.float32),
            base.astype(mx.float32),
            norm_weight.astype(mx.bfloat16),
            params,
        ],
        template=[("HC", hc), ("ITERS", iters), ("MAXPER", -(-dim // 256))],
        grid=(rows * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[
            (batch, length, hc),
            (batch, length, hc),
            (batch, length, hc, hc),
            (batch, length, dim),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32, mx.bfloat16],
    )
    return pre, post, comb, h


# Rows (batch x length) up to which attention uses the decode-shaped kernel.
_DECODE_ROWS = 16

# ---------------------------------------------------------------------------------------
# Shared Metal helpers: RoPE on the trailing 64 features (interleaved pairs, mx.fast.rope's
# arithmetic) for a value held as one element per lane at d = i * 32 + lane, and the
# FP8 E4M3 / power-of-two-scale rounding of the reference.

_HELPERS = """
inline float dsv41_rope_lane(float x, int d, int dim, int rope_dim, float pos, const device float* periods, uint lane) {
    // x is this lane's element d; its pair partner sits in the neighbouring lane
    float other = simd_shuffle_xor(x, 1);
    int start = dim - rope_dim;
    if (d < start) {
        return x;
    }
    int p = (d - start) / 2;
    float inv_freq = 1.0f / periods[p];
    float theta = pos * inv_freq;
    float c = metal::fast::cos(theta);
    float s = metal::fast::sin(theta);
    if ((d - start) % 2 == 0) {
        return x * c - other * s;
    }
    return other * s + x * c;
}

inline float dsv41_pow2_ceil(float x) {
    uint bits = as_type<uint>(x);
    uint exponent = (bits >> 23) & 0xFF;
    uint rounds_up = (bits & 0x7FFFFF) != 0 ? 1 : 0;
    return as_type<float>((exponent + rounds_up) << 23);
}

inline float dsv41_round_e4m3(float v) {
    // round to the E4M3 grid (3 mantissa bits, subnormals below 2^-6), ties to even
    float a = metal::abs(v);
    float quantum;
    if (a < 0.015625f) {
        quantum = 0.001953125f;  // 2^-9
    } else {
        int e;
        metal::frexp(a, e);      // a = m * 2^e, m in [0.5, 1)
        quantum = metal::ldexp(1.0f, e - 4);
    }
    return metal::rint(v / quantum) * quantum;
}
"""

_ATTENTION_ROPE_SOURCE = """
    constexpr int D = HEAD_DIM;
    constexpr int PER = D / 32;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    int heads = q_shape[2];
    int length = q_shape[1];
    int groups = heads / HB;
    int tg = threadgroup_position_in_grid.x;
    int row = tg / groups;
    int h = (tg % groups) * HB + sg;
    int b = row / length;
    int s_idx = row % length;
    float pos = float(start_pos[0] + s_idx);
    float sc = scale[0];
    float qv[PER];
    const device T* qrow = q + (size_t(row) * heads + h) * D;
    for (int i = 0; i < PER; i++) {
        int d = i * 32 + lane;
        float x = float(qrow[d]);
        // RoPE in float32, rounded to the query dtype (as _apply_rope)
        x = float(T(dsv41_rope_lane(x, d, D, ROPE_DIM, pos, periods, lane)));
        qv[i] = sc * x;
    }
    float m = float(sinks[h]);
    float l = 1.0f;
    float acc[PER];
    for (int i = 0; i < PER; i++) {
        acc[i] = 0.0f;
    }
    int kw = widx_shape[2];
    int wlen = window_shape[1];
    const device int* wi = widx + size_t(row) * kw;
    const device T* wbase = window + size_t(b) * wlen * D;
    for (int j = 0; j < kw; j++) {
        int r = wi[j];
        if (r < 0) {
            continue;
        }
        const device T* krow = wbase + size_t(r) * D;
        float kv[PER];
        float dot = 0.0f;
        for (int i = 0; i < PER; i++) {
            kv[i] = float(krow[i * 32 + lane]);
            dot += qv[i] * kv[i];
        }
        float s = simd_sum(dot);
        if (s > m) {
            float c = metal::precise::exp(m - s);
            l = l * c + 1.0f;
            for (int i = 0; i < PER; i++) {
                acc[i] = acc[i] * c + kv[i];
            }
            m = s;
        } else {
            float p = metal::precise::exp(s - m);
            l += p;
            for (int i = 0; i < PER; i++) {
                acc[i] += p * kv[i];
            }
        }
    }
    // BEGIN_COMP
    {
        int kc = cidx_shape[2];
        int clen = comp_shape[1];
        const device int* ci = cidx + size_t(row) * kc;
        const device T* cbase = comp + size_t(b) * clen * D;
        for (int j = 0; j < kc; j++) {
            int r = ci[j];
            if (r < 0) {
                continue;
            }
            const device T* krow = cbase + size_t(r) * D;
            float kv[PER];
            float dot = 0.0f;
            for (int i = 0; i < PER; i++) {
                kv[i] = float(krow[i * 32 + lane]);
                dot += qv[i] * kv[i];
            }
            float s = simd_sum(dot);
            if (s > m) {
                float c = metal::precise::exp(m - s);
                l = l * c + 1.0f;
                for (int i = 0; i < PER; i++) {
                    acc[i] = acc[i] * c + kv[i];
                }
                m = s;
            } else {
                float p = metal::precise::exp(s - m);
                l += p;
                for (int i = 0; i < PER; i++) {
                    acc[i] += p * kv[i];
                }
            }
        }
    }
    // END_COMP
    device T* orow = out + (size_t(row) * heads + h) * D;
    for (int i = 0; i < PER; i++) {
        int d = i * 32 + lane;
        float o = float(T(acc[i] / l));
        // the inverse RoPE of the output, in float32, rounded to the output dtype
        orow[d] = T(dsv41_rope_lane(o, d, D, ROPE_DIM, -pos, periods, lane));
    }
"""


_ATTENTION_DECODE_SOURCE = """
    // one threadgroup per (query row, head); its 8 simdgroups split the keys, then merge
    constexpr int D = HEAD_DIM;
    constexpr int PER = D / 32;
    constexpr int NSG = 8;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint tid = thread_position_in_threadgroup.x;
    int heads = q_shape[2];
    int length = q_shape[1];
    int tg = threadgroup_position_in_grid.x;
    int row = tg / heads;
    int h = tg % heads;
    int b = row / length;
    int s_idx = row % length;
    float pos = float(start_pos[0] + s_idx);
    float sc = scale[0];
    float qv[PER];
    const device T* qrow = q + (size_t(row) * heads + h) * D;
    for (int i = 0; i < PER; i++) {
        int d = i * 32 + lane;
        float x = float(qrow[d]);
        x = float(T(dsv41_rope_lane(x, d, D, ROPE_DIM, pos, periods, lane)));
        qv[i] = sc * x;
    }
    float m = sg == 0 ? float(sinks[h]) : -INFINITY;
    float l = sg == 0 ? 1.0f : 0.0f;
    float acc[PER];
    for (int i = 0; i < PER; i++) {
        acc[i] = 0.0f;
    }
    int kw = widx_shape[2];
    int wlen = window_shape[1];
    int kc = 0;
    // BEGIN_COMP
    kc = cidx_shape[2];
    // END_COMP
    for (int j = sg; j < kw + kc; j += NSG) {
        int r;
        const device T* krow;
        if (j < kw) {
            r = widx[size_t(row) * kw + j];
            krow = window + (size_t(b) * wlen + r) * D;
        } else {
            // BEGIN_COMP
            r = cidx[size_t(row) * kc + (j - kw)];
            krow = comp + (size_t(b) * comp_shape[1] + r) * D;
            // END_COMP
        }
        if (r < 0) {
            continue;
        }
        float kv[PER];
        float dot = 0.0f;
        for (int i = 0; i < PER; i++) {
            kv[i] = float(krow[i * 32 + lane]);
            dot += qv[i] * kv[i];
        }
        float s = simd_sum(dot);
        if (s > m) {
            float c = metal::precise::exp(m - s);
            l = l * c + 1.0f;
            for (int i = 0; i < PER; i++) {
                acc[i] = acc[i] * c + kv[i];
            }
            m = s;
        } else {
            float p = metal::precise::exp(s - m);
            l += p;
            for (int i = 0; i < PER; i++) {
                acc[i] += p * kv[i];
            }
        }
    }
    threadgroup float tm[NSG];
    threadgroup float tl[NSG];
    threadgroup float tacc[NSG][D];
    if (lane == 0) {
        tm[sg] = m;
        tl[sg] = l;
    }
    for (int i = 0; i < PER; i++) {
        tacc[sg][i * 32 + lane] = acc[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float big = tm[0];
    for (int k = 1; k < NSG; k++) {
        big = max(big, tm[k]);
    }
    float total = 0.0f;
    float w[NSG];
    for (int k = 0; k < NSG; k++) {
        w[k] = tm[k] == -INFINITY ? 0.0f : metal::precise::exp(tm[k] - big);
        total += w[k] * tl[k];
    }
    // each thread finishes D / 256 adjacent outputs (a RoPE pair stays in one thread)
    constexpr int OUT_PER = D / (NSG * 32);
    device T* orow = out + (size_t(row) * heads + h) * D;
    int start = D - ROPE_DIM;
    for (int e = 0; e < OUT_PER; e += 2) {
        int d = tid * OUT_PER + e;
        float o0 = 0.0f;
        float o1 = 0.0f;
        for (int k = 0; k < NSG; k++) {
            o0 += w[k] * tacc[k][d];
            o1 += w[k] * tacc[k][d + 1];
        }
        o0 = float(T(o0 / total));
        o1 = float(T(o1 / total));
        if (d >= start) {
            int p = (d - start) / 2;
            float theta = -pos * (1.0f / periods[p]);
            float c = metal::fast::cos(theta);
            float sn = metal::fast::sin(theta);
            float r0 = o0 * c - o1 * sn;
            float r1 = o0 * sn + o1 * c;
            o0 = r0;
            o1 = r1;
        }
        orow[d] = T(o0);
        orow[d + 1] = T(o1);
    }
"""


def _strip_all(source: str, begin: str, end: str) -> str:
    while begin in source:
        source = _strip(source, begin, end)
    return source


@cache
def _attention_decode_kernel(has_comp: bool) -> Callable[..., list[mx.array]]:
    inputs = ["q", "window", "widx", "sinks", "scale", "periods", "start_pos"]
    if has_comp:
        inputs += ["comp", "cidx"]
    source = _ATTENTION_DECODE_SOURCE
    if not has_comp:
        source = _strip_all(source, "// BEGIN_COMP", "// END_COMP")
    return mx.fast.metal_kernel(
        name=f"dsv41_attention_decode_{'wc' if has_comp else 'w'}",
        input_names=inputs,
        output_names=["out"],
        source=source,
        header=_HELPERS,
    )


@cache
def _attention_rope_kernel(has_comp: bool) -> Callable[..., list[mx.array]]:
    inputs = ["q", "window", "widx", "sinks", "scale", "periods", "start_pos"]
    if has_comp:
        inputs += ["comp", "cidx"]
    source = _ATTENTION_ROPE_SOURCE
    if not has_comp:
        source = _strip(source, "// BEGIN_COMP", "// END_COMP")
    return mx.fast.metal_kernel(
        name=f"dsv41_attention_rope_{'wc' if has_comp else 'w'}",
        input_names=inputs,
        output_names=["out"],
        source=source,
        header=_HELPERS,
    )


def attention_with_rope(
    q: mx.array,
    window: mx.array,
    window_idx: mx.array,
    compressed: mx.array | None,
    compressed_idx: mx.array | None,
    *,
    sinks: mx.array,
    scale: float,
    periods: mx.array,
    rope_dim: int,
    start_pos: int,
) -> mx.array | None:
    """Sparse attention (each query attends to the KV rows its indices pick, K = V, with a
    per-head sink; float32 scores, softmax and sums), with the query's RoPE (positions
    start_pos + s) applied first and the inverse RoPE applied to the output, as the
    attention layer does around it. q is before RoPE; -1 picks nothing. None when the
    kernel does not apply."""
    if not fused_available("attention"):
        return None
    batch, length, heads, dim = q.shape
    if dim % 32 or heads % 8 or window.dtype != q.dtype or rope_dim % 64:
        return None
    if compressed is not None and compressed.dtype != q.dtype:
        return None
    hb = 8
    has_comp = compressed is not None and compressed_idx is not None
    inputs = [
        q,
        window,
        window_idx.astype(mx.int32),
        sinks.astype(mx.float32),
        mx.array([scale], dtype=mx.float32),
        periods.astype(mx.float32),
        mx.array([start_pos], dtype=mx.int32),
    ]
    if has_comp:
        assert compressed is not None and compressed_idx is not None
        inputs += [compressed, compressed_idx.astype(mx.int32)]
    if batch * length <= _DECODE_ROWS and dim % 512 == 0:
        (out,) = _attention_decode_kernel(has_comp)(
            inputs=inputs,
            template=[("T", q.dtype), ("HEAD_DIM", dim), ("ROPE_DIM", rope_dim)],
            grid=(batch * length * heads * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[q.shape],
            output_dtypes=[q.dtype],
        )
        return out
    groups = batch * length * (heads // hb)
    (out,) = _attention_rope_kernel(has_comp)(
        inputs=inputs,
        template=[
            ("T", q.dtype),
            ("HB", hb),
            ("HEAD_DIM", dim),
            ("ROPE_DIM", rope_dim),
        ],
        grid=(groups * 32 * hb, 1, 1),
        threadgroup=(32 * hb, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )
    return out


# The sliding-window KV row: RMS norm (bf16 weight), RoPE on the trailing features, then the
# FP8 round trip with a power-of-two scale per 32 values. One simdgroup per row.
_KV_EPILOGUE_SOURCE = """
    constexpr int D = HEAD_DIM;
    constexpr int PER = D / 32;
    uint lane = thread_index_in_simdgroup;
    int row = threadgroup_position_in_grid.x;
    int length = kv_shape[1];
    float pos = float(start_pos[0] + row % length);
    const device T* x = kv + size_t(row) * D;
    float v[PER];
    float sq = 0.0f;
    for (int i = 0; i < PER; i++) {
        v[i] = float(x[i * 32 + lane]);
        sq += v[i] * v[i];
    }
    sq = simd_sum(sq);
    float inv = metal::precise::rsqrt(sq / float(D) + eps[0]);
    device T* o = out + size_t(row) * D;
    for (int i = 0; i < PER; i++) {
        int d = i * 32 + lane;
        float n = float(T(float(w[d]) * float(T(v[i] * inv))));
        n = float(T(dsv41_rope_lane(n, d, D, ROPE_DIM, pos, periods, lane)));
        // block of 32 = this simdgroup's i-th slice
        float amax = simd_max(metal::abs(n));
        amax = metal::max(amax, 1e-4f);
        float scale = dsv41_pow2_ceil(amax * (1.0f / 448.0f));
        float qv = metal::clamp(n / scale, -448.0f, 448.0f);
        o[d] = T(dsv41_round_e4m3(qv) * scale);
    }
"""


@cache
def _kv_epilogue_kernel() -> Callable[..., list[mx.array]]:
    return mx.fast.metal_kernel(
        name="dsv41_kv_epilogue",
        input_names=["kv", "w", "eps", "periods", "start_pos"],
        output_names=["out"],
        source=_KV_EPILOGUE_SOURCE,
        header=_HELPERS,
    )


def kv_epilogue(
    kv: mx.array,
    norm_weight: mx.array,
    *,
    eps: float,
    periods: mx.array,
    rope_dim: int,
    start_pos: int,
) -> mx.array | None:
    """fake_quant_fp8(rope(rms_norm(kv))) for kv [B, L, D] (bf16), positions start_pos + l.
    None when the kernel does not apply."""
    if (
        not fused_available("kv_epilogue")
        or kv.dtype != mx.bfloat16
        or kv.shape[-1] % 32
    ):
        return None
    batch, length, dim = kv.shape
    (out,) = _kv_epilogue_kernel()(
        inputs=[
            kv,
            norm_weight.astype(mx.bfloat16),
            mx.array([eps], dtype=mx.float32),
            periods.astype(mx.float32),
            mx.array([start_pos], dtype=mx.int32),
        ],
        template=[("T", kv.dtype), ("HEAD_DIM", dim), ("ROPE_DIM", rope_dim)],
        grid=(batch * length * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[kv.shape],
        output_dtypes=[kv.dtype],
    )
    return out


# ---------------------------------------------------------------------------------------
# MoE pieces

# Router epilogue: scores = sqrt(softplus(logits / temp)); the top k of scores + bias; their
# unbiased scores normalized and scaled. One simdgroup per row.
_ROUTE_SOURCE = """
    uint lane = thread_index_in_simdgroup;
    int row = threadgroup_position_in_grid.x;
    int n = logits_shape[1];
    constexpr int PER = MAX_E / 32;
    float sc[PER];
    float biased[PER];
    float temp = params[0];
    for (int i = 0; i < PER; i++) {
        int e = i * 32 + lane;
        if (e < n) {
            float x = logits[size_t(row) * n + e] / temp;
            float maxval = metal::max(x, 0.0f);
            float minval = metal::min(x, 0.0f);
            float sp = (maxval == INFINITY) ? maxval : (maxval + log1p(metal::exp(minval - maxval)));
            float s = metal::precise::sqrt(sp);
            sc[i] = s;
            biased[i] = s + bias[e];
        } else {
            sc[i] = 0.0f;
            biased[i] = -INFINITY;
        }
    }
    float w[TOPK];
    int ids[TOPK];
    float total = 0.0f;
    for (int k = 0; k < TOPK; k++) {
        // the largest remaining biased score, lowest expert id on ties
        float best = -INFINITY;
        int best_id = 1 << 30;
        for (int i = 0; i < PER; i++) {
            int e = i * 32 + lane;
            if (biased[i] > best || (biased[i] == best && e < best_id)) {
                best = biased[i];
                best_id = e;
            }
        }
        float top = simd_max(best);
        int cand = (best == top) ? best_id : (1 << 30);
        int chosen = simd_min(cand);
        float chosen_score = 0.0f;
        for (int i = 0; i < PER; i++) {
            int e = i * 32 + lane;
            if (e == chosen) {
                chosen_score = sc[i];
                biased[i] = -INFINITY;
            }
        }
        chosen_score = simd_sum(chosen_score);
        w[k] = chosen_score;
        ids[k] = chosen;
        total += chosen_score;
    }
    if (lane == 0) {
        for (int k = 0; k < TOPK; k++) {
            float weight = NORM ? w[k] / (total + 1e-20f) : w[k];
            weights_out[row * TOPK + k] = weight * params[1];
            ids_out[row * TOPK + k] = uint(ids[k]);
        }
    }
"""


@cache
def _route_kernel() -> Callable[..., list[mx.array]]:
    return mx.fast.metal_kernel(
        name="dsv41_route",
        input_names=["logits", "bias", "params"],
        output_names=["ids_out", "weights_out"],
        source=_ROUTE_SOURCE,
    )


def route(
    logits: mx.array,
    bias: mx.array,
    *,
    top_k: int,
    gate_temp: float,
    norm_topk_prob: bool,
    route_scale: float,
) -> tuple[mx.array, mx.array] | None:
    """The sqrtsoftplus router's selection from float32 logits [R, E]: (ids [R, k] uint32,
    weights [R, k] float32). None when the kernel does not apply."""
    if not fused_available("route") or logits.ndim != 2 or logits.shape[1] > 512:
        return None
    rows, experts = logits.shape
    max_e = 32 * -(-experts // 32)
    ids, weights = _route_kernel()(
        inputs=[
            logits.astype(mx.float32),
            bias.astype(mx.float32),
            mx.array([gate_temp, route_scale], dtype=mx.float32),
        ],
        template=[
            ("MAX_E", max_e),
            ("TOPK", top_k),
            ("NORM", norm_topk_prob and top_k > 1),
        ],
        grid=(rows * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(rows, top_k), (rows, top_k)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return ids, weights


# Clamped SwiGLU: silu(min(gate, limit)) * clip(up, -limit, limit) in float32, rounded to
# the inputs' dtype.
_SWIGLU_SOURCE = """
    uint i = thread_position_in_grid.x;
    if (int(i) >= n[0]) {
        return;
    }
    float limit = params[0];
    float g = float(gate[i]);
    float u = float(up[i]);
    if (limit > 0) {
        g = metal::min(g, limit);
        u = metal::clamp(u, -limit, limit);
    }
    float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(g)));
    float sig = g < 0 ? y : 1.0f - y;
    out[i] = T((g * sig) * u);
"""


@cache
def _swiglu_kernel() -> Callable[..., list[mx.array]]:
    return mx.fast.metal_kernel(
        name="dsv41_swiglu",
        input_names=["up", "gate", "params", "n"],
        output_names=["out"],
        source=_SWIGLU_SOURCE,
    )


def swiglu(up: mx.array, gate: mx.array, limit: float) -> mx.array | None:
    if (
        not fused_available("swiglu")
        or up.shape != gate.shape
        or up.dtype != gate.dtype
    ):
        return None
    size = up.size
    (out,) = _swiglu_kernel()(
        inputs=[
            up,
            gate,
            mx.array([limit], dtype=mx.float32),
            mx.array([size], dtype=mx.int32),
        ],
        template=[("T", up.dtype)],
        grid=(size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[up.shape],
        output_dtypes=[up.dtype],
    )
    return out


# sum_k routed[r, k] * weights[r, k] + shared[r], in float32 (the experts' and the shared
# expert's outputs upcast), in OUT dtype.
_COMBINE_SOURCE = """
    uint idx = thread_position_in_grid.x;
    int dim = shared_shape[1];
    int rows = shared_shape[0];
    if (int(idx) >= rows * dim) {
        return;
    }
    int r = idx / dim;
    int d = idx % dim;
    int k = weights_shape[1];
    float s = 0.0f;
    for (int j = 0; j < k; j++) {
        s += float(routed[(size_t(r) * k + j) * dim + d]) * weights[r * k + j];
    }
    out[idx] = OUT(s + float(shared[idx]));
"""


@cache
def _combine_kernel() -> Callable[..., list[mx.array]]:
    return mx.fast.metal_kernel(
        name="dsv41_moe_combine",
        input_names=["routed", "weights", "shared"],
        output_names=["out"],
        source=_COMBINE_SOURCE,
    )


def moe_combine(
    routed: mx.array, weights: mx.array, shared: mx.array, out_dtype: mx.Dtype
) -> mx.array | None:
    """routed [R, k, D], weights [R, k] (float32), shared [R, D] -> [R, D] in out_dtype."""
    if not fused_available("combine"):
        return None
    rows, dim = shared.shape
    (out,) = _combine_kernel()(
        inputs=[routed, weights.astype(mx.float32), shared],
        template=[("OUT", out_dtype)],
        grid=(rows * dim, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[shared.shape],
        output_dtypes=[out_dtype],
    )
    return out
