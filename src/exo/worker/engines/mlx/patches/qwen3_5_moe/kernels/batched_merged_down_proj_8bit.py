"""Batched merged 8-bit down_proj GEMV for Qwen3.5 routed + shared experts.

Adapts merged_down_proj_8bit for batch_size B (1..8).

Grid z-dimension: B * n_active + 1
  - tgid.z < B * n_active: routed experts (one TG per batch×expert pair)
    Same structure as affine_gather_qmv — each TG independently indexes into
    expert weights via inds[flat_idx].
  - tgid.z == B * n_active: shared expert (ONE TG handles ALL B batch elements
    with register-level weight sharing — loads weights once, computes B outputs)

All constants baked into Metal source (no scalar kernel inputs).
"""
import mlx.core as mx


def ceil_div(a, b):
    return (a + b - 1) // b


def _gen_batched_merged_down_8bit_source(K_OUT, N_IN, SHARED_N_IN, n_active, B, group_size=64):
    gs = int(group_size)
    sc_stride = 256 // gs
    slid_divisor = gs // 8
    N_groups = N_IN // gs
    SHARED_N_groups = SHARED_N_IN // gs
    total_routed = B * n_active

    # Shared expert: generate unrolled batch loops
    shared_x_load_lines = []
    for b in range(B):
        shared_x_load_lines.append(f"""
            float x{b}_thread[VALUES_PER_THREAD];
            float xsum{b} = 0;
            for (int i = 0; i < VALUES_PER_THREAD; i++) {{
                float xi = X_shared[{b} * SHARED_N_IN + x_base + i];
                x{b}_thread[i] = xi;
                xsum{b} += xi;
            }}""")
    shared_x_load = "\n".join(shared_x_load_lines)

    shared_qdot_lines = []
    for b in range(B):
        shared_qdot_lines.append(f"""
                float accum{b} = 0;
                for (int i = 0; i < VALUES_PER_THREAD; i++) {{
                    accum{b} += x{b}_thread[i] * w_vals[i];
                }}
                result{b}[row] += s_val * accum{b} + xsum{b} * b_val;""")
    shared_qdot = "\n".join(shared_qdot_lines)

    shared_result_decls = "\n        ".join(
        f"float result{b}[RESULTS_PER_SG] = {{0, 0, 0, 0}};"
        for b in range(B)
    )

    shared_write_lines = []
    for b in range(B):
        shared_write_lines.append(f"""
        for (int row = 0; row < RESULTS_PER_SG; row++) {{
            float r{b} = simd_sum(result{b}[row]);
            if (slid == 0) {{
                Y_shared[{b} * K_OUT + out_row + row] = r{b};
            }}
        }}""")
    shared_write = "\n".join(shared_write_lines)

    return f"""
    const int RESULTS_PER_SG = 4;
    const int VALUES_PER_THREAD = 8;
    const int BLOCK_SIZE = 256;
    const int K_OUT = {K_OUT};
    const int N_IN = {N_IN};
    const int SHARED_N_IN = {SHARED_N_IN};
    const int N_GROUPS = {N_groups};
    const int SHARED_N_GROUPS = {SHARED_N_groups};
    const int N_ACTIVE = {n_active};
    const int TOTAL_ROUTED = {total_routed};
    const int BATCH_SIZE = {B};

    uint3 tgid = threadgroup_position_in_grid;
    uint sgid = simdgroup_index_in_threadgroup;
    uint slid = thread_index_in_simdgroup;

    int out_row = tgid.y * 8 + sgid * RESULTS_PER_SG;
    if (out_row >= K_OUT) return;

    if (tgid.z < (uint)TOTAL_ROUTED) {{
        // ═══════ ROUTED EXPERT PATH (same as gather_qmv) ═══════
        // tgid.z indexes flat (batch, expert) pairs
        int flat_idx = (int)tgid.z;
        int expert = inds[flat_idx];

        const device uint8_t* ws = (const device uint8_t*)W
            + (long)expert * K_OUT * N_IN + out_row * N_IN + slid * VALUES_PER_THREAD;
        const device bfloat16_t* sc = (const device bfloat16_t*)S
            + (long)expert * K_OUT * N_GROUPS + out_row * N_GROUPS + slid / {slid_divisor};
        const device bfloat16_t* bi = (const device bfloat16_t*)B_q
            + (long)expert * K_OUT * N_GROUPS + out_row * N_GROUPS + slid / {slid_divisor};

        const device float* x_ptr = (const device float*)X_routed
            + flat_idx * N_IN;

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
                float s = float(sc[row * N_GROUPS]);
                float b = float(bi[row * N_GROUPS]);
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

        device float* yp = Y_routed + flat_idx * K_OUT + out_row;
        for (int row = 0; row < RESULTS_PER_SG; row++) {{
            float r = simd_sum(result[row]);
            if (slid == 0) {{
                yp[row] = r;
            }}
        }}

    }} else {{
        // ═══════ SHARED EXPERT PATH (register-level weight sharing) ═══════
        // ONE TG handles ALL {B} batch elements.
        // Load shared expert weights once, compute {B} outputs from registers.

        const device uint8_t* ws = (const device uint8_t*)W_shared_down
            + (long)out_row * SHARED_N_IN + slid * VALUES_PER_THREAD;
        const device bfloat16_t* sc = (const device bfloat16_t*)S_shared_down
            + (long)out_row * SHARED_N_GROUPS + slid / {slid_divisor};
        const device bfloat16_t* bi = (const device bfloat16_t*)B_shared_down
            + (long)out_row * SHARED_N_GROUPS + slid / {slid_divisor};

        int x_base = slid * VALUES_PER_THREAD;
        {shared_result_decls}

        for (int k = 0; k < SHARED_N_IN; k += BLOCK_SIZE) {{
            // Load x for all {B} batch elements
{shared_x_load}

            // Load weights once into registers, compute all {B} batches
            for (int row = 0; row < RESULTS_PER_SG; row++) {{
                const device uint8_t* wl = ws + row * SHARED_N_IN;
                float s_val = float(sc[row * SHARED_N_GROUPS]);
                float b_val = float(bi[row * SHARED_N_GROUPS]);

                float w_vals[VALUES_PER_THREAD];
                for (int i = 0; i < VALUES_PER_THREAD; i++) {{
                    w_vals[i] = float(wl[i]);
                }}

{shared_qdot}
            }}

            ws += BLOCK_SIZE;
            sc += {sc_stride};
            bi += {sc_stride};
            x_base += BLOCK_SIZE;
        }}

        // Write all {B} outputs
{shared_write}
    }}
"""


_batched_down_cache = {}


def _get_batched_down_kernel(K_OUT, N_IN, SHARED_N_IN, n_active, B, group_size=64):
    key = (K_OUT, N_IN, SHARED_N_IN, n_active, B, group_size)
    if key not in _batched_down_cache:
        _batched_down_cache[key] = mx.fast.metal_kernel(
            name=f"batched_down_K{K_OUT}_N{N_IN}_SN{SHARED_N_IN}_na{n_active}_B{B}",
            input_names=["W", "S", "B_q",
                         "W_shared_down", "S_shared_down", "B_shared_down",
                         "X_routed", "X_shared", "inds"],
            output_names=["Y_routed", "Y_shared"],
            source=_gen_batched_merged_down_8bit_source(
                K_OUT, N_IN, SHARED_N_IN, n_active, B, group_size),
        )
    return _batched_down_cache[key]


def batched_merged_down_proj_8bit(w_q, s, b_q,
                                   w_shared_down, s_shared_down, b_shared_down,
                                   x_routed, x_shared, inds,
                                   k_out, n_in, batch_size,
                                   n_active, group_size=64,
                                   shared_n_in=None):
    """Batched merged down_proj for 8-bit routed + shared experts.

    Args:
        w_q: routed weights (E, K_OUT, N_IN/4) uint32
        s: routed scales (E, K_OUT, N_IN/gs) bf16
        b_q: routed biases (E, K_OUT, N_IN/gs) bf16
        w_shared_down: shared weight (K_OUT, SHARED_N_IN/4) uint32
        s_shared_down: shared scales (K_OUT, SHARED_N_IN/gs) bf16
        b_shared_down: shared biases (K_OUT, SHARED_N_IN/gs) bf16
        x_routed: (B * n_active, N_IN) f32
        x_shared: (B * SHARED_N_IN,) f32
        inds: (B * n_active,) uint32
        k_out: output dimension
        n_in: routed input dimension
        batch_size: B
        n_active: experts per token (top_k)
        shared_n_in: shared input dim (defaults to n_in)

    Returns:
        Y_routed: (B * n_active, k_out) f32
        Y_shared: (B, k_out) f32
    """
    B = batch_size
    k_out_val = int(k_out)
    n_in_val = int(n_in)
    shared_n_in_val = int(shared_n_in) if shared_n_in is not None else n_in_val

    kern = _get_batched_down_kernel(k_out_val, n_in_val, shared_n_in_val, n_active, B)

    y_groups = ceil_div(k_out_val, 8)
    total_routed = B * n_active

    Y = kern(
        inputs=[w_q, s, b_q,
                w_shared_down, s_shared_down, b_shared_down,
                x_routed, x_shared, inds],
        output_shapes=[(total_routed * k_out_val,), (B * k_out_val,)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(32, y_groups * 2, total_routed + 1),
        threadgroup=(32, 2, 1),
    )

    return Y[0].reshape(total_routed, k_out_val), Y[1].reshape(B, k_out_val)
