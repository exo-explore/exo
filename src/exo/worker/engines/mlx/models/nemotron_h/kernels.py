"""Metal kernels for Nemotron-H: the Mamba-2 mixer between `in_proj` and `out_proj`, the
MoE router's expert selection and the MoE output combination.

They compute what model.py's float32 path computes (NVIDIA's reference precision), in
fewer GPU dispatches:

- `causal_conv_silu`: the depthwise causal conv over the conv state and the new inputs,
  its bias and SiLU, in float32 (one kernel), plus the next conv state.
- `selective_scan`: the Mamba-2 recurrence in float32 for any number of tokens, with the
  time step (softplus of `dt + dt_bias`, clipped), the `D` skip connection and the SiLU
  gate fused in (one kernel). Each SIMD group keeps its slice of the state in registers
  and walks the tokens in order, so prefill no longer builds the chunked (L x L)
  matrices of mlx-lm's `ssm_attn`.
- `grouped_rms_norm`: the gated RMSNorm's per-group normalisation and weight, rounded
  once to the model's dtype for `out_proj` (one kernel).
- `select_experts`: sigmoid, score-correction bias, top-k, normalisation and scaling of
  float32 router logits (one kernel, `n_group == 1`).
- `combine_experts`: the routed experts' weighted sum in float32, rounded to the model
  dtype, plus the shared expert and the block's residual, each rounded as mlx-lm does.

Each has a reference implementation in MLX ops (`*_reference`) that the tests compare
against, and model.py falls back to mlx-lm's ops where a kernel does not apply (no Metal,
masked or variable-length batches, grouped routing).
"""

from functools import cache

import mlx.core as mx
import mlx.nn as nn

_CONV_SOURCE = """
    // proj: (B, L, P) T, conv input at columns [OFF, OFF + C); state: (B, K - 1, C) T;
    // weight: (C, K) (MLX's depthwise conv layout (C, K, 1)); bias: (C,).
    // Rows r < L: xbc[b, r, c] = silu(sum_k weight[c, k] * padded[r + k] + bias[c]) in
    // float32, padded = concat(state, input). Rows L .. L + K - 2: the next conv state, the
    // last K - 1 rows of padded.
    uint c = thread_position_in_grid.x;
    uint r = thread_position_in_grid.y;
    uint b = thread_position_in_grid.z;
    if (c >= C) {
        return;
    }
    const int L = proj_shape[1];
    const int P = proj_shape[2];
    constexpr int KM1 = K - 1;
    if ((int)r < L) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            const int p = (int)r + k;
            const float v = p < KM1
                ? static_cast<float>(state[((size_t)b * KM1 + p) * C + c])
                : static_cast<float>(proj[((size_t)b * L + (p - KM1)) * P + OFF + c]);
            acc += static_cast<float>(weight[c * K + k]) * v;
        }
        if (HAS_BIAS) {
            acc += static_cast<float>(bias[c]);
        }
        xbc[((size_t)b * L + r) * C + c] = acc / (1.0f + metal::exp(-acc));
    } else {
        const int j = (int)r - L;
        const int p = L + j;
        state_out[((size_t)b * KM1 + j) * C + c] = p < KM1
            ? state[((size_t)b * KM1 + p) * C + c]
            : proj[((size_t)b * L + (p - KM1)) * P + OFF + c];
    }
"""

_SCAN_SOURCE = """
    // xbc: (B, L, CD) float = [x: H * Dh | B: NG * Ds | C: NG * Ds] (after conv + SiLU);
    // proj: (B, L, P) T with the gate at columns [0, H * Dh) and dt at [DT_OFF, DT_OFF + H);
    // dt_bias, A_log, D: (H,); limits: (2,) float; state_in: (B, H, Dh, Ds) float.
    // y: (B, L, H * Dh) float = (scan + D * x) * silu(gate); state_out like state_in.
    // A SIMD group owns DPS rows (d) of one head's state, Ds / 32 columns per lane, and
    // walks the L tokens in order.
    constexpr int NPT = Ds / 32;
    constexpr int IS = H * Dh;
    constexpr int CD = IS + 2 * NG * Ds;
    const uint lane = thread_index_in_simdgroup;
    const uint d0 = thread_position_in_grid.y * DPS;
    const uint n = thread_position_in_grid.z;
    const uint h = n % H;
    const uint b = n / H;
    const uint g = h / (H / NG);
    const int L = xbc_shape[1];
    const int P = proj_shape[2];

    float st[DPS][NPT];
    for (int j = 0; j < DPS; ++j) {
        const size_t base = ((size_t)n * Dh + d0 + j) * Ds + NPT * lane;
        for (int i = 0; i < NPT; ++i) {
            st[j][i] = state_in[base + i];
        }
    }
    const float A = -metal::precise::exp(static_cast<float>(A_log[h]));
    const float Dv = static_cast<float>(D[h]);
    const float bias = static_cast<float>(dt_bias[h]);
    const float lo = limits[0];
    const float hi = limits[1];

    for (int t = 0; t < L; ++t) {
        const size_t row = (size_t)b * L + t;
        // softplus(dt + dt_bias) = max(v, 0) + log1p(exp(-|v|)), as mlx's logaddexp(v, 0)
        const float v = static_cast<float>(proj[row * P + DT_OFF + h]) + bias;
        const float e = metal::precise::exp(-metal::abs(v));
        const float u = 1.0f + e;
        const float log1p_e = u == 1.0f ? e : metal::precise::log(u) * (e / (u - 1.0f));
        const float dt = metal::clamp(metal::max(v, 0.0f) + log1p_e, lo, hi);
        const float dA = metal::precise::exp(A * dt);

        const device float* xrow = xbc + row * CD;
        const device float* Bp = xrow + IS + g * Ds + NPT * lane;
        const device float* Cp = xrow + IS + NG * Ds + g * Ds + NPT * lane;
        float bv[NPT];
        float cv[NPT];
        for (int i = 0; i < NPT; ++i) {
            bv[i] = Bp[i];
            cv[i] = Cp[i];
        }
        float out_j = 0.0f;
        for (int j = 0; j < DPS; ++j) {
            const float x = xrow[h * Dh + d0 + j];
            const float xdt = x * dt;
            float acc = 0.0f;
            for (int i = 0; i < NPT; ++i) {
                st[j][i] = dA * st[j][i] + xdt * bv[i];
                acc += st[j][i] * cv[i];
            }
            acc = simd_sum(acc);
            if (lane == (uint)j) {
                out_j = acc + x * Dv;
            }
        }
        if (lane < DPS) {
            const uint col = h * Dh + d0 + lane;
            const float z = static_cast<float>(proj[row * P + col]);
            y[row * IS + col] = out_j * (z / (1.0f + metal::exp(-z)));
        }
    }
    for (int j = 0; j < DPS; ++j) {
        const size_t base = ((size_t)n * Dh + d0 + j) * Ds + NPT * lane;
        for (int i = 0; i < NPT; ++i) {
            state_out[base + i] = st[j][i];
        }
    }
"""

_NORM_SOURCE = """
    // y: (R, GS) float, R = rows * NG; weight: (NG * GS,); eps: (1,) float.
    // out[r, i] = weight[g * GS + i] * (y[r, i] * rsqrt(mean(y[r]^2) + eps)), rounded once.
    const uint r = threadgroup_position_in_grid.y;
    const uint tid = thread_position_in_threadgroup.x;
    const uint g = r % NG;
    const device float* v = y + (size_t)r * GS;
    float ss = 0.0f;
    for (int i = tid; i < GS; i += TG) {
        const float a = v[i];
        ss += a * a;
    }
    threadgroup float partial[32];
    ss = simd_sum(ss);
    if (thread_index_in_simdgroup == 0) {
        partial[simdgroup_index_in_threadgroup] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (int i = 0; i < TG / 32; ++i) {
        total += partial[i];
    }
    const float scale = metal::precise::rsqrt(total / GS + eps[0]);
    for (int i = tid; i < GS; i += TG) {
        out[(size_t)r * GS + i] =
            static_cast<OUT_T>(static_cast<float>(weight[g * GS + i]) * (v[i] * scale));
    }
"""

_SELECT_SOURCE = """
    // logits: (N, E) float; bias: (E,) float; scaling: (1,) float.
    // inds: (N, K + EXTRA) uint32, scores: (N, K + EXTRA) float. One SIMD group per row:
    // sigmoid, add the correction bias, take the K largest (ties to the lower expert),
    // gather their sigmoid scores, normalise them (NORM) and scale.
    constexpr int VPT = (E + 31) / 32;
    const uint row = thread_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup;
    float orig[VPT];
    float sel[VPT];
    for (int i = 0; i < VPT; ++i) {
        const int e = lane + 32 * i;
        if (e < E) {
            const float l = logits[(size_t)row * E + e];
            const float s = 1.0f / (1.0f + metal::exp(-l));
            orig[i] = s;
            sel[i] = s + static_cast<float>(bias[e]);
        } else {
            orig[i] = 0.0f;
            sel[i] = -INFINITY;
        }
    }
    float chosen_scores[K];
    uint chosen_experts[K];
    float total = 0.0f;
    for (int k = 0; k < K; ++k) {
        float best = -INFINITY;
        int best_e = E;
        for (int i = 0; i < VPT; ++i) {
            if (sel[i] > best) {
                best = sel[i];
                best_e = lane + 32 * i;
            }
        }
        const float top = simd_max(best);
        const int expert = simd_min(best == top ? best_e : E);
        float score = 0.0f;
        if ((int)lane == expert % 32) {
            score = orig[expert / 32];
            sel[expert / 32] = -INFINITY;
        }
        score = simd_sum(score);
        total += score;
        chosen_scores[k] = score;
        chosen_experts[k] = (uint)expert;
    }
    if (lane == 0) {
        const float denominator = total + 1e-20f;
        constexpr int KO = K + EXTRA;
        for (int k = 0; k < K; ++k) {
            const float s = (NORM && K > 1) ? chosen_scores[k] / denominator : chosen_scores[k];
            inds[(size_t)row * KO + k] = chosen_experts[k];
            scores[(size_t)row * KO + k] = s * scaling[0];
        }
        // Experts E .. E + EXTRA - 1 (a shared expert folded into the routed ones) always
        // take part, with weight 1.
        for (int j = 0; j < EXTRA; ++j) {
            inds[(size_t)row * KO + K + j] = (uint)(E + j);
            scores[(size_t)row * KO + K + j] = 1.0f;
        }
    }
"""

_COMBINE_SOURCE = """
    // y: (N, K, Dm) T expert outputs; scores: (N, K) float; shared: (N, Dm) T;
    // residual: (N, Dm) T. out = residual + (round(sum_k scores * y) + shared), with each
    // sum rounded to T as mlx-lm's (y * scores).sum().astype(T), + shared, x + h do.
    const uint i = thread_position_in_grid.x;
    const uint n = thread_position_in_grid.y;
    if (i >= Dm) {
        return;
    }
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += static_cast<float>(y[((size_t)n * K + k) * Dm + i]) * scores[(size_t)n * K + k];
    }
    T h = static_cast<T>(acc);
    if (HAS_SHARED) {
        h = static_cast<T>(static_cast<float>(h) + static_cast<float>(shared[(size_t)n * Dm + i]));
    }
    if (HAS_RESIDUAL) {
        h = static_cast<T>(static_cast<float>(residual[(size_t)n * Dm + i]) + static_cast<float>(h));
    }
    out[(size_t)n * Dm + i] = h;
"""


_RESIDUAL_NORM_SOURCE = """
    // Row r: y: (R, K, Dm) T [HAS_Y]; scores: (R, K) float [HAS_SCORES]; shared: (R, Dm) T
    // [HAS_SHARED]; residual: (R, Dm) T; weight: (Dm,) T; eps: (1,) float.
    // hidden = residual [+ round(round(sum_k scores_k * y_k) [+ shared])], each step rounded
    // to T as mlx-lm's (y * scores).sum().astype(T) + shared and x + h do;
    // normed = weight * round(hidden * rsqrt(mean(hidden^2) + eps)) (MLX's RMSNorm rounding);
    // normed32 = float(normed) [F32]. The reduction order depends only on Dm and TG, so
    // every RMSNorm computed with this kernel rounds the same way.
    constexpr int PER = (Dm + TG - 1) / TG;
    const uint r = threadgroup_position_in_grid.y;
    const uint tid = thread_position_in_threadgroup.x;
    T hv[PER];
    float ss = 0.0f;
    for (int j = 0; j < PER; ++j) {
        const int i = tid + j * TG;
        if (i < Dm) {
            T h = residual[(size_t)r * Dm + i];
            if (HAS_Y) {
                float acc = 0.0f;
                for (int k = 0; k < K; ++k) {
                    const float yv = static_cast<float>(y[((size_t)r * K + k) * Dm + i]);
                    acc += HAS_SCORES ? yv * scores[(size_t)r * K + k] : yv;
                }
                T m = static_cast<T>(acc);
                if (HAS_SHARED) {
                    m = static_cast<T>(
                        static_cast<float>(m) + static_cast<float>(shared[(size_t)r * Dm + i]));
                }
                h = static_cast<T>(static_cast<float>(h) + static_cast<float>(m));
                hidden[(size_t)r * Dm + i] = h;
            }
            hv[j] = h;
            const float hf = static_cast<float>(h);
            ss += hf * hf;
        }
    }
    threadgroup float partial[32];
    ss = simd_sum(ss);
    if (thread_index_in_simdgroup == 0) {
        partial[simdgroup_index_in_threadgroup] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (int i = 0; i < (TG + 31) / 32; ++i) {
        total += partial[i];
    }
    const float inv = metal::precise::rsqrt(total / Dm + eps[0]);
    for (int j = 0; j < PER; ++j) {
        const int i = tid + j * TG;
        if (i < Dm) {
            const T n = static_cast<T>(
                static_cast<float>(weight[i])
                * static_cast<float>(static_cast<T>(static_cast<float>(hv[j]) * inv)));
            normed[(size_t)r * Dm + i] = n;
            if (F32) {
                normed32[(size_t)r * Dm + i] = static_cast<float>(n);
            }
        }
    }
"""


_CONSTANTS: dict[tuple[str, int], mx.array] = {}


def _zeros(dtype: mx.Dtype) -> mx.array:
    """A one-element placeholder input, made once (an unused input of a kernel)."""
    key = (str(dtype), 1)
    if key not in _CONSTANTS:
        placeholder = mx.zeros((1,), dtype=dtype)
        mx.eval(placeholder)
        _CONSTANTS[key] = placeholder
    return _CONSTANTS[key]


def metal_available() -> bool:
    return mx.metal.is_available() and mx.default_device() == mx.gpu


@cache
def _kernels() -> dict[str, "mx.fast.MetalKernel"]:
    return {
        "conv": mx.fast.metal_kernel(
            name="nemotron_h_causal_conv_silu",
            input_names=["proj", "state", "weight", "bias"],
            output_names=["xbc", "state_out"],
            source=_CONV_SOURCE,
        ),
        "scan": mx.fast.metal_kernel(
            name="nemotron_h_selective_scan",
            input_names=["xbc", "proj", "dt_bias", "A_log", "D", "limits", "state_in"],
            output_names=["y", "state_out"],
            source=_SCAN_SOURCE,
        ),
        "norm": mx.fast.metal_kernel(
            name="nemotron_h_grouped_rms_norm",
            input_names=["y", "weight", "eps"],
            output_names=["out"],
            source=_NORM_SOURCE,
        ),
        "select": mx.fast.metal_kernel(
            name="nemotron_h_select_experts",
            input_names=["logits", "bias", "scaling"],
            output_names=["inds", "scores"],
            source=_SELECT_SOURCE,
        ),
        "combine": mx.fast.metal_kernel(
            name="nemotron_h_combine_experts",
            input_names=["y", "scores", "shared", "residual"],
            output_names=["out"],
            source=_COMBINE_SOURCE,
        ),
        "residual_norm": mx.fast.metal_kernel(
            name="nemotron_h_residual_rms_norm",
            input_names=["y", "scores", "shared", "residual", "weight", "eps"],
            output_names=["hidden", "normed", "normed32"],
            source=_RESIDUAL_NORM_SOURCE,
        ),
    }


def causal_conv_silu(
    proj: mx.array,
    offset: int,
    channels: int,
    state: mx.array,
    weight: mx.array,
    bias: mx.array | None,
) -> tuple[mx.array, mx.array]:
    """Depthwise causal conv + bias + SiLU in float32 over `proj[..., offset:offset +
    channels]` preceded by `state` (B, K - 1, channels). Returns the float32 output
    (B, L, channels) and the next state (the last K - 1 inputs, in `state`'s dtype)."""
    batch, length, _ = proj.shape
    kernel_size = state.shape[1] + 1
    kernel = _kernels()["conv"]
    threads = min(256, channels)
    outputs = kernel(
        inputs=[
            proj,
            state,
            weight.reshape(channels, kernel_size),
            bias if bias is not None else _zeros(proj.dtype),
        ],
        template=[
            ("C", channels),
            ("K", kernel_size),
            ("OFF", offset),
            ("HAS_BIAS", bias is not None),
        ],
        grid=(channels, length + kernel_size - 1, batch),
        threadgroup=(threads, 1, 1),
        output_shapes=[(batch, length, channels), state.shape],
        output_dtypes=[mx.float32, state.dtype],
    )
    return outputs[0], outputs[1]


def selective_scan(
    xbc: mx.array,
    proj: mx.array,
    dt_offset: int,
    dt_bias: mx.array,
    a_log: mx.array,
    d_skip: mx.array,
    limits: tuple[float, float],
    state: mx.array,
    num_heads: int,
    head_dim: int,
    n_groups: int,
    state_size: int,
) -> tuple[mx.array, mx.array]:
    """The Mamba-2 recurrence over `xbc` (B, L, [x | B | C]) from `state`
    (B, H, Dh, Ds) float32, with the time step taken from `proj[..., dt_offset:]`, the D
    skip and the SiLU gate (`proj[..., :H * Dh]`). Returns the gated output (B, L, H * Dh)
    float32 and the final state."""
    batch, length, _ = xbc.shape
    rows_per_simdgroup = 1 if length < 16 else 4
    if head_dim % rows_per_simdgroup:
        rows_per_simdgroup = 1
    simdgroups_y = head_dim // rows_per_simdgroup
    threadgroup_y = next(t for t in (8, 4, 2, 1) if simdgroups_y % t == 0)
    kernel = _kernels()["scan"]
    outputs = kernel(
        inputs=[
            xbc,
            proj,
            dt_bias,
            a_log,
            d_skip,
            mx.array(limits, dtype=mx.float32),
            state,
        ],
        template=[
            ("T", proj.dtype),
            ("H", num_heads),
            ("Dh", head_dim),
            ("Ds", state_size),
            ("NG", n_groups),
            ("DT_OFF", dt_offset),
            ("DPS", rows_per_simdgroup),
        ],
        grid=(32, simdgroups_y, num_heads * batch),
        threadgroup=(32, threadgroup_y, 1),
        output_shapes=[(batch, length, num_heads * head_dim), state.shape],
        output_dtypes=[mx.float32, mx.float32],
    )
    return outputs[0], outputs[1]


def grouped_rms_norm(
    y: mx.array, weight: mx.array, group_size: int, eps: float, dtype: mx.Dtype
) -> mx.array:
    """RMSNorm of float32 `y` over groups of `group_size` channels, times `weight`,
    rounded once to `dtype`."""
    n_groups = y.shape[-1] // group_size
    rows = y.size // group_size
    threads = min(1024, max(32, group_size // 2))
    kernel = _kernels()["norm"]
    outputs = kernel(
        inputs=[y, weight, mx.array([eps], dtype=mx.float32)],
        template=[
            ("GS", group_size),
            ("NG", n_groups),
            ("TG", threads),
            ("OUT_T", dtype),
        ],
        grid=(threads, rows, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[y.shape],
        output_dtypes=[dtype],
    )
    return outputs[0]


def select_experts(
    logits: mx.array,
    bias: mx.array,
    top_k: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
    always_on: int = 0,
) -> tuple[mx.array, mx.array]:
    """`group_expert_select` for `n_group == 1` on float32 logits (..., E), followed by
    `always_on` experts E, E + 1, ... with weight 1 (see `with_always_on_experts`)."""
    *lead, n_experts = logits.shape
    rows = logits.size // n_experts
    kernel = _kernels()["select"]
    outputs = kernel(
        inputs=[
            logits,
            bias.astype(mx.float32),
            mx.array([routed_scaling_factor], dtype=mx.float32),
        ],
        template=[
            ("E", n_experts),
            ("K", top_k),
            ("NORM", norm_topk_prob),
            ("EXTRA", always_on),
        ],
        grid=(32, rows, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(*lead, top_k + always_on), (*lead, top_k + always_on)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return outputs[0], outputs[1]


def combine_experts(
    y: mx.array,
    scores: mx.array,
    shared: mx.array | None,
    residual: mx.array | None,
) -> mx.array:
    """round(sum_k scores * y) [+ shared] [residual +], each step rounded to y's dtype."""
    *lead, top_k, dims = y.shape
    rows = y.size // (top_k * dims)
    threads = min(256, dims)
    dummy = _zeros(y.dtype)
    kernel = _kernels()["combine"]
    outputs = kernel(
        inputs=[
            y,
            scores.astype(mx.float32),
            shared if shared is not None else dummy,
            residual if residual is not None else dummy,
        ],
        template=[
            ("T", y.dtype),
            ("K", top_k),
            ("Dm", dims),
            ("HAS_SHARED", shared is not None),
            ("HAS_RESIDUAL", residual is not None),
        ],
        grid=(dims, rows, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(*lead, dims)],
        output_dtypes=[y.dtype],
    )
    return outputs[0]


def norm_threads(dims: int) -> int:
    """Threads per row of `residual_norm`: a function of the row size only, so that every
    RMSNorm of a model reduces in the same order."""
    return min(1024, max(32, -(-dims // 4 // 32) * 32))


def residual_norm(
    residual: mx.array,
    weight: mx.array,
    eps: float,
    y: mx.array | None = None,
    scores: mx.array | None = None,
    shared: mx.array | None = None,
    want_float32: bool = False,
) -> tuple[mx.array, mx.array, mx.array | None]:
    """A block boundary in one kernel: `hidden = residual + block output` (the block
    output is `y` (..., Dm), or the MoE's `y` (..., K, Dm) weighted by `scores` plus
    `shared`), then the next RMSNorm of `hidden`. Returns (hidden, normed, normed as
    float32 if `want_float32`). Without `y` it is the RMSNorm of `residual` alone."""
    dims = residual.shape[-1]
    rows = residual.size // dims
    has_scores = scores is not None
    if y is not None and not has_scores:
        y = mx.expand_dims(y, -2)
    top_k = 1 if y is None else y.shape[-2]
    threads = norm_threads(dims)
    dtype = residual.dtype
    kernel = _kernels()["residual_norm"]
    outputs = kernel(
        inputs=[
            y if y is not None else _zeros(dtype),
            scores.astype(mx.float32) if scores is not None else _zeros(mx.float32),
            shared if shared is not None else _zeros(dtype),
            residual,
            weight,
            mx.array([eps], dtype=mx.float32),
        ],
        template=[
            ("T", dtype),
            ("Dm", dims),
            ("K", top_k),
            ("TG", threads),
            ("HAS_Y", y is not None),
            ("HAS_SCORES", has_scores),
            ("HAS_SHARED", shared is not None),
            ("F32", want_float32),
        ],
        grid=(threads, rows, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[
            residual.shape if y is not None else (1,),
            residual.shape,
            residual.shape if want_float32 else (1,),
        ],
        output_dtypes=[dtype, dtype, mx.float32],
    )
    hidden = outputs[0] if y is not None else residual
    return hidden, outputs[1], outputs[2] if want_float32 else None


def with_always_on_experts(
    inds: mx.array, scores: mx.array, n_experts: int, always_on: int
) -> tuple[mx.array, mx.array]:
    """Append experts n_experts, ..., n_experts + always_on - 1 with weight 1 to each
    row's selection (the MLX-op form of `select_experts`'s `always_on`)."""
    if always_on == 0:
        return inds, scores
    lead = inds.shape[:-1]
    extra = mx.broadcast_to(
        mx.arange(n_experts, n_experts + always_on, dtype=inds.dtype),
        (*lead, always_on),
    )
    ones = mx.ones((*lead, always_on), dtype=scores.dtype)
    return mx.concatenate([inds, extra], axis=-1), mx.concatenate([scores, ones], -1)


# References in MLX ops: what model.py computed before these kernels.


def causal_conv_silu_reference(
    proj: mx.array,
    offset: int,
    channels: int,
    state: mx.array,
    weight: mx.array,
    bias: mx.array | None,
) -> tuple[mx.array, mx.array]:
    conv_input = proj[..., offset : offset + channels]
    padded = mx.concatenate([state, conv_input], axis=1)
    kernel_size = state.shape[1] + 1
    y = mx.conv1d(
        padded.astype(mx.float32),
        weight.reshape(channels, kernel_size, 1).astype(mx.float32),
        groups=channels,
    )
    if bias is not None:
        y = y + bias.astype(mx.float32)
    return nn.silu(y), padded[:, -(kernel_size - 1) :, :]
