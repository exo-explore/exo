"""Merged 8-bit down_proj GEMV for Qwen3.5 routed + shared experts.

Port of the 4-bit down_proj kernel, adapted for 8-bit quantization (gs=64).
Maps intermediate → hidden: (n_active, N_IN) → (n_active, K_OUT).

Two paths via tgid.z:
- 8-bit routed experts (z < n_active): uint8 dequant with expert index lookup, f32 output
- 8-bit shared expert (z == n_active): uint8 dequant (no expert index), f32 output
"""
import mlx.core as mx


def ceil_div(a, b):
    return (a + b - 1) // b


def _gen_merged_down_8bit_source(group_size=64, scale_bf16=True):
    """Metal source for merged 8-bit down_proj GEMV.

    Same structure as gate GEMV: 2 SGs × 4 rows/SG = 8 output rows per TG.
    8-bit dequant: result = scale * Σ(x[i]*w[i]) + bias * Σ(x[i])
    Both routed and shared paths use 8-bit dequantization.
    """
    gs = int(group_size)
    sc_stride = 256 // gs  # groups per K-block (256/64 = 4)
    slid_divisor = gs // 8  # threads per group (64/8 = 8)
    sc_t = "bfloat16_t" if scale_bf16 else "float"

    return f"""
    const int RESULTS_PER_SG = 4;
    const int VALUES_PER_THREAD = 8;
    const int BLOCK_SIZE = 256;

    int K_OUT = K_OUT_val;
    int N_IN = N_IN_val;
    int SHARED_N_IN = SHARED_N_IN_val;
    int n_active = n_active_val;

    uint3 tgid = threadgroup_position_in_grid;
    uint sgid = simdgroup_index_in_threadgroup;  // 0 or 1
    uint slid = thread_index_in_simdgroup;       // 0..31

    int out_row = tgid.y * 8 + sgid * RESULTS_PER_SG;
    if (out_row >= K_OUT) return;

    if (tgid.z < (uint)n_active) {{
        // ═══════ 8-BIT ROUTED EXPERT PATH ═══════
        int N_groups = N_IN / {gs};

        int expert = inds[tgid.z];

        const device uint8_t* ws = (const device uint8_t*)W
            + (long)expert * K_OUT * N_IN + out_row * N_IN + slid * VALUES_PER_THREAD;
        const device {sc_t}* sc = (const device {sc_t}*)S
            + (long)expert * K_OUT * N_groups + out_row * N_groups + slid / {slid_divisor};
        const device {sc_t}* bi = (const device {sc_t}*)B_q
            + (long)expert * K_OUT * N_groups + out_row * N_groups + slid / {slid_divisor};

        const device float* x_ptr = (const device float*)X_routed
            + tgid.z * N_IN;

        int x_base = slid * VALUES_PER_THREAD;
        float result[4] = {{0, 0, 0, 0}};

        for (int k = 0; k < N_IN; k += BLOCK_SIZE) {{
            float x_thread[8];
            float xsum = 0;
            for (int i = 0; i < 8; i++) {{
                float xi = x_ptr[x_base + i];
                x_thread[i] = xi;
                xsum += xi;
            }}

            for (int row = 0; row < RESULTS_PER_SG; row++) {{
                const device uint8_t* wl = ws + row * N_IN;
                float s = float(sc[row * N_groups]);
                float b = float(bi[row * N_groups]);
                float accum = 0;
                for (int i = 0; i < 8; i++) {{
                    accum += x_thread[i] * float(wl[i]);
                }}
                result[row] += s * accum + xsum * b;
            }}

            ws += BLOCK_SIZE;
            sc += {sc_stride};
            bi += {sc_stride};
            x_base += BLOCK_SIZE;
        }}

        device float* yp = Y_routed + tgid.z * K_OUT + out_row;
        for (int row = 0; row < RESULTS_PER_SG; row++) {{
            float r = simd_sum(result[row]);
            if (slid == 0) {{
                yp[row] = r;
            }}
        }}

    }} else {{
        // ═══════ 8-BIT SHARED EXPERT PATH ═══════
        // Same dequant as routed path, but no expert index lookup.
        // W_shared_down is (K_OUT, SHARED_N_IN/4) uint32 → (K_OUT, SHARED_N_IN) uint8
        int N_groups = SHARED_N_IN / {gs};

        const device uint8_t* ws = (const device uint8_t*)W_shared_down
            + (long)out_row * SHARED_N_IN + slid * VALUES_PER_THREAD;
        const device {sc_t}* sc = (const device {sc_t}*)S_shared_down
            + (long)out_row * N_groups + slid / {slid_divisor};
        const device {sc_t}* bi = (const device {sc_t}*)B_shared_down
            + (long)out_row * N_groups + slid / {slid_divisor};

        // X_shared is float32 (output of Kernel 1 shared path)
        const device float* x_ptr = (const device float*)X_shared;

        int x_base = slid * VALUES_PER_THREAD;
        float result[4] = {{0, 0, 0, 0}};

        for (int k = 0; k < SHARED_N_IN; k += BLOCK_SIZE) {{
            float x_thread[8];
            float xsum = 0;
            for (int i = 0; i < 8; i++) {{
                float xi = x_ptr[x_base + i];
                x_thread[i] = xi;
                xsum += xi;
            }}

            for (int row = 0; row < RESULTS_PER_SG; row++) {{
                const device uint8_t* wl = ws + row * SHARED_N_IN;
                float s = float(sc[row * N_groups]);
                float b = float(bi[row * N_groups]);
                float accum = 0;
                for (int i = 0; i < 8; i++) {{
                    accum += x_thread[i] * float(wl[i]);
                }}
                result[row] += s * accum + xsum * b;
            }}

            ws += BLOCK_SIZE;
            sc += {sc_stride};
            bi += {sc_stride};
            x_base += BLOCK_SIZE;
        }}

        device float* yp = Y_shared + out_row;
        for (int tm = 0; tm < RESULTS_PER_SG; tm++) {{
            float r = simd_sum(result[tm]);
            if (slid == 0) {{
                yp[tm] = r;
            }}
        }}
    }}
"""


_merged_down_8bit_kernels = {}


def _get_merged_down_8bit_kernel(group_size=64, scale_bf16=True):
    key = (group_size, scale_bf16)
    if key not in _merged_down_8bit_kernels:
        sc_tag = "_bf16sc" if scale_bf16 else ""
        _merged_down_8bit_kernels[key] = mx.fast.metal_kernel(
            name=f"merged_down_proj_8bit_gs{group_size}{sc_tag}",
            input_names=["W", "S", "B_q",
                         "W_shared_down", "S_shared_down", "B_shared_down",
                         "X_routed", "X_shared", "inds",
                         "K_OUT_val", "N_IN_val", "SHARED_N_IN_val", "n_active_val"],
            output_names=["Y_routed", "Y_shared"],
            source=_gen_merged_down_8bit_source(group_size, scale_bf16),
        )
    return _merged_down_8bit_kernels[key]


def fused_merged_down_proj_8bit(w_q, s, b_q,
                                 w_shared_down, s_shared_down, b_shared_down,
                                 x_routed, x_shared, inds,
                                 k_out, n_in, group_size=64,
                                 shared_n_in=None):
    """Single-dispatch merged down_proj for 8-bit routed + 8-bit shared experts.

    Args:
        w_q: routed quantized down weights (E, K_OUT, N_IN/4) uint32
        s: routed scales (E, K_OUT, N_IN/gs) bfloat16
        b_q: routed biases (E, K_OUT, N_IN/gs) bfloat16
        w_shared_down: shared expert down weight (K_OUT, SHARED_N_IN/4) uint32
        s_shared_down: shared expert down scales (K_OUT, SHARED_N_IN/gs) bfloat16
        b_shared_down: shared expert down biases (K_OUT, SHARED_N_IN/gs) bfloat16
        x_routed: routed SwiGLU output (n_active, N_IN) float32
        x_shared: shared SwiGLU output (SHARED_N_IN,) float32
        inds: selected expert indices (n_active,) uint32
        k_out: output dimension (4096 for Qwen3.5)
        n_in: routed expert input dimension (1024 for Qwen3.5)
        group_size: quantization group size (64)
        shared_n_in: shared expert input dimension (defaults to n_in)

    Returns:
        (Y_routed, Y_shared):
            Y_routed: (n_active, k_out) float32
            Y_shared: (k_out,) float32
    """
    scale_bf16 = (s.dtype == mx.bfloat16)
    kern = _get_merged_down_8bit_kernel(group_size, scale_bf16)
    n_active = inds.shape[0]
    k_out_val = int(k_out)
    n_in_val = int(n_in)
    shared_n_in_val = int(shared_n_in) if shared_n_in is not None else n_in_val
    y_groups = ceil_div(k_out_val, 8)
    Y = kern(
        inputs=[w_q, s, b_q,
                w_shared_down, s_shared_down, b_shared_down,
                x_routed, x_shared, inds,
                k_out_val, n_in_val, shared_n_in_val, n_active],
        output_shapes=[(n_active, k_out_val), (k_out_val,)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(32, y_groups * 2, n_active + 1),
        threadgroup=(32, 2, 1),
    )
    return Y[0], Y[1]
