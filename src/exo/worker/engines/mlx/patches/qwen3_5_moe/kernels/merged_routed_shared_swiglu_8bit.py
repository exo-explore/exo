"""Merged 8-bit fused gate+up+SwiGLU for Qwen3.5 routed + shared experts.

Port of the 4-bit kernel for Kimi K2.5, adapted for:
- 8-bit quantization (direct uint8 byte reads, no nibble extraction)
- group_size=64 (one scale+bias per 64 elements)
- Qwen3.5 dimensions (K=4096, N_INTER=1024, E=512, top_k=10)
- No score normalization in kernel (handled by gate dispatch)

Two paths via tgid.z:
- 8-bit routed experts (z < n_active): uint8 dequant with expert index lookup, f32 output
- 8-bit shared expert (z == n_active): uint8 dequant (no expert index), f32 output
"""
import mlx.core as mx


def ceil_div(a, b):
    return (a + b - 1) // b


def _gen_merged_8bit_source(group_size=64, scale_bf16=True):
    """Generate Metal source for merged 8-bit fused gate+up+SwiGLU.

    Both routed and shared expert paths use 8-bit dequant:
        result = scale * Σ(x[i]*w[i]) + bias * Σ(x[i])

    Each thread processes VALUES_PER_THREAD=8 elements of K per iteration.
    BLOCK_SIZE = 32 * 8 = 256 elements per K-block.
    """
    gs = int(group_size)
    sc_stride = 256 // gs  # groups consumed per K-block (256/64 = 4)
    slid_divisor = gs // 8  # threads per group (64/8 = 8)
    sc_t = "bfloat16_t" if scale_bf16 else "float"

    return f"""
    const int RESULTS_PER_SG = 4;

    int N_INTER = N_INTER_val;
    int SHARED_INTER = SHARED_INTER_val;
    int K = K_val;
    int n_active = n_active_val;

    uint3 tgid = threadgroup_position_in_grid;
    uint sgid = simdgroup_index_in_threadgroup;  // 0 or 1
    uint slid = thread_index_in_simdgroup;       // 0..31

    int out_row = tgid.y * 8 + sgid * RESULTS_PER_SG;

    // Routed path uses N_INTER rows, shared path uses SHARED_INTER rows
    int row_limit = (tgid.z < (uint)n_active) ? N_INTER : SHARED_INTER;
    if (out_row >= row_limit) return;

    float gate_result[4] = {{0, 0, 0, 0}};
    float up_result[4] = {{0, 0, 0, 0}};

    if (tgid.z < (uint)n_active) {{
        // ═══════ 8-BIT ROUTED EXPERT PATH ═══════
        const int VALUES_PER_THREAD = 8;
        const int BLOCK_SIZE = 256;  // 32 * 8

        int N_TOTAL = 2 * N_INTER;  // gate + up stacked
        int K_groups = K / {gs};

        int expert = inds[tgid.z];

        // W is stored as uint32 (4 bytes per uint32), cast to uint8_t
        // Layout: (E, N_TOTAL, K/4) uint32 → (E, N_TOTAL, K) uint8
        const device uint8_t* ws_gate = (const device uint8_t*)W
            + (long)expert * N_TOTAL * K + out_row * K + slid * VALUES_PER_THREAD;
        const device {sc_t}* sc_gate = (const device {sc_t}*)S
            + (long)expert * N_TOTAL * K_groups + out_row * K_groups + slid / {slid_divisor};
        const device {sc_t}* bi_gate = (const device {sc_t}*)B_q
            + (long)expert * N_TOTAL * K_groups + out_row * K_groups + slid / {slid_divisor};

        const device uint8_t* ws_up = (const device uint8_t*)W
            + (long)expert * N_TOTAL * K + (out_row + N_INTER) * K + slid * VALUES_PER_THREAD;
        const device {sc_t}* sc_up = (const device {sc_t}*)S
            + (long)expert * N_TOTAL * K_groups + (out_row + N_INTER) * K_groups + slid / {slid_divisor};
        const device {sc_t}* bi_up = (const device {sc_t}*)B_q
            + (long)expert * N_TOTAL * K_groups + (out_row + N_INTER) * K_groups + slid / {slid_divisor};

        int x_base = slid * VALUES_PER_THREAD;

        for (int k = 0; k < K; k += BLOCK_SIZE) {{
            // Load 8 x values and compute sum
            float x_thread[8];
            float xsum = 0;
            for (int i = 0; i < 8; i++) {{
                float xi = float(X[x_base + i]);
                x_thread[i] = xi;
                xsum += xi;
            }}

            for (int row = 0; row < RESULTS_PER_SG; row++) {{
                // Gate projection
                const device uint8_t* wg = ws_gate + row * K;
                float sg = float(sc_gate[row * K_groups]);
                float bg = float(bi_gate[row * K_groups]);
                float accum_g = 0;
                for (int i = 0; i < 8; i++) {{
                    accum_g += x_thread[i] * float(wg[i]);
                }}
                gate_result[row] += sg * accum_g + xsum * bg;

                // Up projection
                const device uint8_t* wu = ws_up + row * K;
                float su = float(sc_up[row * K_groups]);
                float bu = float(bi_up[row * K_groups]);
                float accum_u = 0;
                for (int i = 0; i < 8; i++) {{
                    accum_u += x_thread[i] * float(wu[i]);
                }}
                up_result[row] += su * accum_u + xsum * bu;
            }}

            ws_gate += BLOCK_SIZE;
            ws_up += BLOCK_SIZE;
            sc_gate += {sc_stride};
            sc_up += {sc_stride};
            bi_gate += {sc_stride};
            bi_up += {sc_stride};
            x_base += BLOCK_SIZE;
        }}

        // Epilogue: SwiGLU + write f32 to Y_routed
        device float* yp = Y_routed + tgid.z * N_INTER + out_row;
        for (int row = 0; row < RESULTS_PER_SG; row++) {{
            float g = simd_sum(gate_result[row]);
            float u = simd_sum(up_result[row]);
            if (slid == 0) {{
                float silu_g = g / (1.0f + metal::exp(-g));
                yp[row] = silu_g * u;
            }}
        }}

    }} else {{
        // ═══════ 8-BIT SHARED EXPERT PATH ═══════
        // Same dequant as routed path, but no expert index lookup.
        // W_shared is (2*SHARED_INTER, K/4) uint32 → (2*SHARED_INTER, K) uint8
        const int VALUES_PER_THREAD = 8;
        const int BLOCK_SIZE = 256;  // 32 * 8

        int K_groups = K / {gs};

        const device uint8_t* ws_gate = (const device uint8_t*)W_shared
            + (long)out_row * K + slid * VALUES_PER_THREAD;
        const device {sc_t}* sc_gate = (const device {sc_t}*)S_shared
            + (long)out_row * K_groups + slid / {slid_divisor};
        const device {sc_t}* bi_gate = (const device {sc_t}*)B_shared
            + (long)out_row * K_groups + slid / {slid_divisor};

        const device uint8_t* ws_up = (const device uint8_t*)W_shared
            + (long)(out_row + SHARED_INTER) * K + slid * VALUES_PER_THREAD;
        const device {sc_t}* sc_up = (const device {sc_t}*)S_shared
            + (long)(out_row + SHARED_INTER) * K_groups + slid / {slid_divisor};
        const device {sc_t}* bi_up = (const device {sc_t}*)B_shared
            + (long)(out_row + SHARED_INTER) * K_groups + slid / {slid_divisor};

        int x_base = slid * VALUES_PER_THREAD;

        for (int k = 0; k < K; k += BLOCK_SIZE) {{
            float x_thread[8];
            float xsum = 0;
            for (int i = 0; i < 8; i++) {{
                float xi = float(X[x_base + i]);
                x_thread[i] = xi;
                xsum += xi;
            }}

            for (int row = 0; row < RESULTS_PER_SG; row++) {{
                // Gate projection
                const device uint8_t* wg = ws_gate + row * K;
                float sg = float(sc_gate[row * K_groups]);
                float bg = float(bi_gate[row * K_groups]);
                float accum_g = 0;
                for (int i = 0; i < 8; i++) {{
                    accum_g += x_thread[i] * float(wg[i]);
                }}
                gate_result[row] += sg * accum_g + xsum * bg;

                // Up projection
                const device uint8_t* wu = ws_up + row * K;
                float su = float(sc_up[row * K_groups]);
                float bu = float(bi_up[row * K_groups]);
                float accum_u = 0;
                for (int i = 0; i < 8; i++) {{
                    accum_u += x_thread[i] * float(wu[i]);
                }}
                up_result[row] += su * accum_u + xsum * bu;
            }}

            ws_gate += BLOCK_SIZE;
            ws_up += BLOCK_SIZE;
            sc_gate += {sc_stride};
            sc_up += {sc_stride};
            bi_gate += {sc_stride};
            bi_up += {sc_stride};
            x_base += BLOCK_SIZE;
        }}

        // Epilogue: SwiGLU + write f32 to Y_shared
        device float* yp = Y_shared + out_row;
        for (int tm = 0; tm < RESULTS_PER_SG; tm++) {{
            float g = simd_sum(gate_result[tm]);
            float u = simd_sum(up_result[tm]);
            if (slid == 0) {{
                float silu_g = g / (1.0f + metal::exp(-g));
                yp[tm] = silu_g * u;
            }}
        }}
    }}
"""


_merged_8bit_kernels = {}


def _get_merged_8bit_kernel(group_size=64, scale_bf16=True):
    key = (group_size, scale_bf16)
    if key not in _merged_8bit_kernels:
        sc_tag = "_bf16sc" if scale_bf16 else ""
        _merged_8bit_kernels[key] = mx.fast.metal_kernel(
            name=f"merged_routed_shared_swiglu_8bit_gs{group_size}{sc_tag}",
            input_names=["W", "S", "B_q", "W_shared", "S_shared", "B_shared",
                         "X", "inds",
                         "N_INTER_val", "SHARED_INTER_val", "K_val", "n_active_val"],
            output_names=["Y_routed", "Y_shared"],
            source=_gen_merged_8bit_source(group_size, scale_bf16),
        )
    return _merged_8bit_kernels[key]


def fused_merged_gate_up_swiglu_8bit(w_q, s, b_q,
                                      w_shared, s_shared, b_shared,
                                      x, inds,
                                      n_inter, k_hidden, group_size=64,
                                      shared_inter=None):
    """Single-dispatch merged gate+up+SwiGLU for 8-bit routed + 8-bit shared experts.

    Args:
        w_q: stacked routed quantized weights (E, 2*N_INTER, K/4) uint32
        s: routed scales (E, 2*N_INTER, K/gs) bfloat16
        b_q: routed biases (E, 2*N_INTER, K/gs) bfloat16
        w_shared: shared expert gate+up stacked (2*SHARED_INTER, K/4) uint32
        s_shared: shared expert scales (2*SHARED_INTER, K/gs) bfloat16
        b_shared: shared expert biases (2*SHARED_INTER, K/gs) bfloat16
        x: input vector (K,) bfloat16
        inds: selected expert indices (n_active,) uint32
        n_inter: routed expert intermediate size (1024 for Qwen3.5)
        k_hidden: hidden size (4096 for Qwen3.5)
        group_size: quantization group size (64)
        shared_inter: shared expert intermediate size (defaults to n_inter)

    Returns:
        (Y_routed, Y_shared):
            Y_routed: (n_active, n_inter) float32
            Y_shared: (shared_inter,) float32
    """
    scale_bf16 = (s.dtype == mx.bfloat16)
    kern = _get_merged_8bit_kernel(group_size, scale_bf16)
    n_active = inds.shape[0]
    n_inter_val = int(n_inter)
    shared_inter_val = int(shared_inter) if shared_inter is not None else n_inter_val
    # Grid y must cover max(n_inter, shared_inter)
    max_inter = max(n_inter_val, shared_inter_val)
    Y = kern(
        inputs=[w_q, s, b_q, w_shared, s_shared, b_shared,
                x, inds,
                n_inter_val, shared_inter_val, int(k_hidden), n_active],
        output_shapes=[(n_active, n_inter_val), (shared_inter_val,)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(32, ceil_div(max_inter, 8) * 2, n_active + 1),
        threadgroup=(32, 2, 1),
    )
    return Y[0], Y[1]
