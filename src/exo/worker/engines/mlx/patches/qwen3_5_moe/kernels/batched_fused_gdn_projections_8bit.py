"""Batched fused GDN projections for Qwen3.5-35B-A3B, batch_size 1..8.

Register-level weight sharing: each TG loads weights once, computes B outputs.
Adapts fused_gdn_projections_8bit with the same pattern as
batched_fused_gqa_projections_8bit.

4 regions with different epilogues:
  - QKV: GEMV → conv1d(4-tap) → SiLU → bf16 + cache update
  - Z:   GEMV → SiLU → f32
  - B:   GEMV → sigmoid → f32 (beta for GDN kernel)
  - A:   GEMV → g=exp(-exp(A_log)*softplus(a+dt_bias)) → f32

All constants baked into Metal source. B unrolled at code-generation time.

Grid: (32, total_tg * 2, 1), TG: (32, 2, 1)
No grid z for batch — batch is handled in registers.
"""

import mlx.core as mx

from ..common import COMPUTE_DTYPE, METAL_HALF_TYPE


def ceil_div(a, b):
    return (a + b - 1) // b


def _gen_batched_fused_gdn_proj_source(K, N_QKV, N_Z, N_B, N_A, B, group_size=64):
    gs = int(group_size)
    sc_stride = 256 // gs
    slid_div = gs // 8
    N_TOTAL = N_QKV + N_Z + N_B + N_A
    K_groups = K // gs
    N_QKV_TG = ceil_div(N_QKV, 8)
    N_Z_TG = ceil_div(N_Z, 8)
    N_B_TG = ceil_div(N_B, 8)

    # Per-batch x loading (B unrolled)
    x_load = "\n".join(f"""
        float x{b}_thread[VALUES_PER_THREAD]; float xsum{b} = 0;
        for (int i = 0; i < VALUES_PER_THREAD; i++) {{
            float xi = float(x[{b} * K + x_base + i]); x{b}_thread[i] = xi; xsum{b} += xi;
        }}""" for b in range(B))

    # Per-batch dot product with weights in registers
    qdot = "\n".join(f"""
            float accum{b} = 0;
            for (int i = 0; i < VALUES_PER_THREAD; i++) accum{b} += x{b}_thread[i] * w_vals[i];
            result{b}[row] += s_val * accum{b} + xsum{b} * b_val;""" for b in range(B))

    result_decls = " ".join(f"float result{b}[4] = {{0,0,0,0}};" for b in range(B))
    simd_reduce = "\n    ".join(
        f"for (int row = 0; row < 4; row++) result{b}[row] = simd_sum(result{b}[row]);" for b in range(B))

    # QKV epilogue: conv1d + SiLU + cache update, per batch
    qkv_write = "\n".join(f"""
            if (slid < 4u && c < N_QKV) {{
                float qkv_val = result{b}[slid];
                long cs_base = (long){b} * 3 * conv_dim;
                float s0 = float(conv_state[cs_base + 0 * conv_dim + c]);
                float s1 = float(conv_state[cs_base + 1 * conv_dim + c]);
                float s2 = float(conv_state[cs_base + 2 * conv_dim + c]);
                float conv_out = float(conv_w[c * 4 + 0]) * s0
                               + float(conv_w[c * 4 + 1]) * s1
                               + float(conv_w[c * 4 + 2]) * s2
                               + float(conv_w[c * 4 + 3]) * qkv_val;
                float silu_out = conv_out / (1.0f + metal::exp(-conv_out));
                conv_state_out[cs_base + 0 * conv_dim + c] = static_cast<bfloat16_t>(s1);
                conv_state_out[cs_base + 1 * conv_dim + c] = static_cast<bfloat16_t>(s2);
                conv_state_out[cs_base + 2 * conv_dim + c] = static_cast<bfloat16_t>(qkv_val);
                qkv_out[{b} * conv_dim + c] = static_cast<bfloat16_t>(silu_out);
            }}""" for b in range(B))

    # Z epilogue: SiLU per batch
    z_write = "\n".join(f"""
            if (slid < 4u && z_row < N_Z) {{
                float val = result{b}[slid];
                z_silu_out[{b} * N_Z + z_row] = val / (1.0f + metal::exp(-val));
            }}""" for b in range(B))

    # B epilogue: sigmoid per batch
    b_write = "\n".join(f"""
            if (slid < 4u && b_row < N_B) {{
                b_out[{b} * N_B + b_row] = 1.0f / (1.0f + metal::exp(-result{b}[slid]));
            }}""" for b in range(B))

    # A epilogue: g computation per batch
    a_write = "\n".join(f"""
            if (slid < 4u && a_row < N_A_val) {{
                float a_val = result{b}[slid];
                float dt = float(dt_bias_arr[a_row]);
                float x_g = a_val + dt;
                float sp = (x_g > 20.0f) ? x_g : metal::log(1.0f + metal::exp(x_g));
                float g_val = metal::exp(-metal::exp(float(A_log_arr[a_row])) * sp);
                a_out[{b} * N_A_val + a_row] = g_val;
            }}""" for b in range(B))

    return f"""
    const int RESULTS_PER_SG = 4;
    const int VALUES_PER_THREAD = 8;
    const int BLOCK_SIZE = 256;
    const int GROUP_SIZE = {gs};
    const int SC_STRIDE = {sc_stride};
    const int SLID_DIV = {slid_div};
    const int K = {K};
    const int K_groups = {K_groups};
    const int N_QKV = {N_QKV};
    const int N_Z = {N_Z};
    const int N_B = {N_B};
    const int N_TOTAL = {N_TOTAL};
    const int N_QKV_TG = {N_QKV_TG};
    const int N_Z_TG = {N_Z_TG};
    const int N_B_TG = {N_B_TG};

    uint3 tgid = threadgroup_position_in_grid;
    uint sgid = simdgroup_index_in_threadgroup;
    uint slid = thread_index_in_simdgroup;
    int tg = tgid.y;

    int out_row, region;
    if (tg < N_QKV_TG) {{
        region = 0; out_row = tg * 8 + sgid * RESULTS_PER_SG;
    }} else if (tg < N_QKV_TG + N_Z_TG) {{
        region = 1; out_row = N_QKV + (tg - N_QKV_TG) * 8 + sgid * RESULTS_PER_SG;
    }} else if (tg < N_QKV_TG + N_Z_TG + N_B_TG) {{
        region = 2; out_row = N_QKV + N_Z + (tg - N_QKV_TG - N_Z_TG) * 8 + sgid * RESULTS_PER_SG;
    }} else {{
        region = 3; out_row = N_QKV + N_Z + N_B + (tg - N_QKV_TG - N_Z_TG - N_B_TG) * 8 + sgid * RESULTS_PER_SG;
    }}
    if (out_row >= N_TOTAL) return;

    // Weight pointers (shared across all batch elements)
    const device uint8_t* ws = (const device uint8_t*)W_merged + (long)out_row * K + slid * VALUES_PER_THREAD;
    const device bfloat16_t* sc = (const device bfloat16_t*)S_merged + (long)out_row * K_groups + slid / SLID_DIV;
    const device bfloat16_t* bi = (const device bfloat16_t*)B_merged + (long)out_row * K_groups + slid / SLID_DIV;

    {result_decls}
    int x_base = slid * VALUES_PER_THREAD;

    // K-loop: load weights into registers once, compute {B} batch elements
    for (int k_off = 0; k_off < K; k_off += BLOCK_SIZE) {{
{x_load}

        for (int row = 0; row < RESULTS_PER_SG; row++) {{
            const device uint8_t* wl = ws + row * K;
            float s_val = float(sc[row * K_groups]);
            float b_val = float(bi[row * K_groups]);
            float w_vals[VALUES_PER_THREAD];
            for (int i = 0; i < VALUES_PER_THREAD; i++) w_vals[i] = float(wl[i]);
{qdot}
        }}

        ws += BLOCK_SIZE; sc += SC_STRIDE; bi += SC_STRIDE; x_base += BLOCK_SIZE;
    }}

    {simd_reduce}

    // Region-specific epilogues for all {B} batches
    if (region == 0) {{
        int c = out_row + (int)slid;
        int conv_dim = N_QKV;
{qkv_write}
    }} else if (region == 1) {{
        int z_row = out_row - N_QKV + (int)slid;
{z_write}
    }} else if (region == 2) {{
        int b_row = out_row - N_QKV - N_Z + (int)slid;
{b_write}
    }} else {{
        int a_row = out_row - N_QKV - N_Z - N_B + (int)slid;
        int N_A_val = N_TOTAL - N_QKV - N_Z - N_B;
{a_write}
    }}
"""


_batched_gdn_proj_cache = {}


def _get_batched_gdn_proj_kernel(K, N_QKV, N_Z, N_B, N_A, B, group_size=64):
    key = (K, N_QKV, N_Z, N_B, N_A, B, group_size)
    if key not in _batched_gdn_proj_cache:
        _batched_gdn_proj_cache[key] = mx.fast.metal_kernel(
            name=f"batched_fused_gdn_proj_K{K}_NQKV{N_QKV}_B{B}",
            input_names=[
                "x",
                "W_merged", "S_merged", "B_merged",
                "conv_state", "conv_w",
                "A_log_arr", "dt_bias_arr",
            ],
            output_names=["qkv_out", "z_silu_out", "b_out", "a_out", "conv_state_out"],
            source=_gen_batched_fused_gdn_proj_source(K, N_QKV, N_Z, N_B, N_A, B, group_size).replace("bfloat16_t", METAL_HALF_TYPE),
        )
    return _batched_gdn_proj_cache[key]


def batched_fused_gdn_projections(
    x,
    W_merged, S_merged, B_merged,
    proj_dims,
    conv_state, conv_weights,
    A_log, dt_bias,
    batch_size=1,
):
    """Batched fused GDN projections with register-level weight sharing.

    Same as fused_gdn_projections but loads weights once per TG and computes
    B outputs from registers. No grid z for batch.

    Args:
        x: [B, 1, K] bf16 — post-RMSNorm hidden state
        W_merged, S_merged, B_merged: merged quantized weights
        proj_dims: (N_QKV, N_Z, N_B, N_A)
        conv_state: [B, 3, conv_dim] bf16
        conv_weights: [conv_dim, 4, 1] or [conv_dim, 4] bf16
        A_log: [Hv] f32, dt_bias: [Hv] f32
        batch_size: int (1..8)

    Returns:
        qkv_conv_silu: [B, 1, N_QKV] bf16
        z_silu:        [B, 1, N_Z] f32
        beta:          [B, 1, N_B] f32
        g:             [B, 1, N_A] f32
        conv_state_out: [B, 3, N_QKV] bf16
    """
    B = batch_size

    N_QKV, N_Z, N_B, N_A = proj_dims
    K = x.shape[-1]

    kern = _get_batched_gdn_proj_kernel(K, N_QKV, N_Z, N_B, N_A, B)

    N_QKV_TG = ceil_div(N_QKV, 8)
    N_Z_TG = ceil_div(N_Z, 8)
    N_B_TG = ceil_div(N_B, 8)
    N_A_TG = ceil_div(N_A, 8)
    total_tg = N_QKV_TG + N_Z_TG + N_B_TG + N_A_TG

    conv_w_flat = conv_weights.reshape(-1, 4) if conv_weights.ndim == 3 else conv_weights
    x_flat = x.reshape(B, K)

    results = kern(
        inputs=[
            x_flat,
            W_merged, S_merged, B_merged,
            conv_state, conv_w_flat,
            A_log, dt_bias,
        ],
        output_shapes=[
            (B * N_QKV,),
            (B * N_Z,),
            (B * N_B,),
            (B * N_A,),
            (B * 3 * N_QKV,),
        ],
        output_dtypes=[COMPUTE_DTYPE, mx.float32, mx.float32, mx.float32, COMPUTE_DTYPE],
        grid=(32, total_tg * 2, 1),   # No grid z — batch in registers
        threadgroup=(32, 2, 1),
    )

    qkv_out = results[0].reshape(B, 1, N_QKV)
    z_silu = results[1].reshape(B, 1, N_Z)
    beta = results[2].reshape(B, 1, N_B)
    g = results[3].reshape(B, 1, N_A)
    conv_state_out = results[4].reshape(B, 3, N_QKV)

    return qkv_out, z_silu, beta, g, conv_state_out
