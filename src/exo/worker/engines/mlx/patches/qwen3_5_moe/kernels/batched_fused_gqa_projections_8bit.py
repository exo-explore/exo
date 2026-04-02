"""Batched fused GQA projections (Dispatch 1) for batch_size 1..8.

Adapts fused_gqa_projections_8bit for B>1 with register-level weight sharing.
Each TG loads weights once, computes B outputs from registers.

4 regions with different epilogues (same as B=1):
  - Queries: GEMV → raw bf16
  - Gate: GEMV → sigmoid → f32
  - Keys: GEMV → raw bf16
  - Values: GEMV → raw bf16

All constants baked into Metal source. B unrolled at code-generation time.
"""

import mlx.core as mx


def ceil_div(a, b):
    return (a + b - 1) // b


def _gen_batched_fused_gqa_proj_source(K, N_Q, N_GATE, N_K, N_V, B, group_size=64):
    gs = int(group_size)
    sc_stride = 256 // gs
    slid_div = gs // 8
    N_TOTAL = N_Q + N_GATE + N_K + N_V
    K_groups = K // gs
    N_Q_TG = ceil_div(N_Q, 8)
    N_GATE_TG = ceil_div(N_GATE, 8)
    N_K_TG = ceil_div(N_K, 8)

    # Per-batch x loading
    x_load = "\n".join(f"""
        float x{b}_thread[VALUES_PER_THREAD]; float xsum{b} = 0;
        for (int i = 0; i < VALUES_PER_THREAD; i++) {{
            float xi = float(x[{b} * K + x_base + i]); x{b}_thread[i] = xi; xsum{b} += xi;
        }}""" for b in range(B))

    # Per-batch qdot (weights in registers)
    qdot = "\n".join(f"""
            float accum{b} = 0;
            for (int i = 0; i < VALUES_PER_THREAD; i++) accum{b} += x{b}_thread[i] * w_vals[i];
            result{b}[row] += s_val * accum{b} + xsum{b} * b_val;""" for b in range(B))

    result_decls = " ".join(f"float result{b}[4] = {{0,0,0,0}};" for b in range(B))
    simd_reduce = "\n    ".join(
        f"for (int row = 0; row < 4; row++) result{b}[row] = simd_sum(result{b}[row]);" for b in range(B))

    # Queries epilogue (bf16 write per batch)
    q_write = "\n".join(f"""
            if (slid < 4u && q_row < N_Q) q_out[{b} * N_Q + q_row] = static_cast<bfloat16_t>(result{b}[slid]);"""
        for b in range(B))

    # Gate epilogue (sigmoid → f32 per batch)
    gate_write = "\n".join(f"""
            if (slid < 4u && g_row < N_GATE) {{
                float sig{b} = 1.0f / (1.0f + metal::exp(-result{b}[slid]));
                gate_out[{b} * N_GATE + g_row] = sig{b};
            }}""" for b in range(B))

    # Keys epilogue
    k_write = "\n".join(f"""
            if (slid < 4u && k_row < N_K) k_out[{b} * N_K + k_row] = static_cast<bfloat16_t>(result{b}[slid]);"""
        for b in range(B))

    # Values epilogue
    v_write = "\n".join(f"""
            if (slid < 4u && v_row < N_V) v_out[{b} * N_V + v_row] = static_cast<bfloat16_t>(result{b}[slid]);"""
        for b in range(B))

    N_V_val = N_TOTAL - N_Q - N_GATE - N_K

    return f"""
    const int RESULTS_PER_SG = 4;
    const int VALUES_PER_THREAD = 8;
    const int BLOCK_SIZE = 256;
    const int GROUP_SIZE = {gs};
    const int SC_STRIDE = {sc_stride};
    const int SLID_DIV = {slid_div};
    const int K = {K};
    const int K_groups = {K_groups};
    const int N_Q = {N_Q};
    const int N_GATE = {N_GATE};
    const int N_K = {N_K};
    const int N_V = {N_V_val};
    const int N_TOTAL = {N_TOTAL};
    const int N_Q_TG = {N_Q_TG};
    const int N_GATE_TG = {N_GATE_TG};
    const int N_K_TG = {N_K_TG};

    uint3 tgid = threadgroup_position_in_grid;
    uint sgid = simdgroup_index_in_threadgroup;
    uint slid = thread_index_in_simdgroup;
    int b_idx = tgid.z;
    int tg = tgid.y;

    int out_row, region;
    if (tg < N_Q_TG) {{
        region = 0; out_row = tg * 8 + sgid * RESULTS_PER_SG;
    }} else if (tg < N_Q_TG + N_GATE_TG) {{
        region = 1; out_row = N_Q + (tg - N_Q_TG) * 8 + sgid * RESULTS_PER_SG;
    }} else if (tg < N_Q_TG + N_GATE_TG + N_K_TG) {{
        region = 2; out_row = N_Q + N_GATE + (tg - N_Q_TG - N_GATE_TG) * 8 + sgid * RESULTS_PER_SG;
    }} else {{
        region = 3; out_row = N_Q + N_GATE + N_K + (tg - N_Q_TG - N_GATE_TG - N_K_TG) * 8 + sgid * RESULTS_PER_SG;
    }}
    if (out_row >= N_TOTAL) return;

    // Weight pointers (shared across all batch elements)
    const device uint8_t* ws = (const device uint8_t*)W_merged + (long)out_row * K + slid * VALUES_PER_THREAD;
    const device bfloat16_t* sc = (const device bfloat16_t*)S_merged + (long)out_row * K_groups + slid / SLID_DIV;
    const device bfloat16_t* bi = (const device bfloat16_t*)B_merged + (long)out_row * K_groups + slid / SLID_DIV;

    {result_decls}
    int x_base = slid * VALUES_PER_THREAD;

    // K-loop: load weights once, compute {B} batch elements
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
        int q_row = out_row + (int)slid;
{q_write}
    }} else if (region == 1) {{
        int g_row = out_row - N_Q + (int)slid;
{gate_write}
    }} else if (region == 2) {{
        int k_row = out_row - N_Q - N_GATE + (int)slid;
{k_write}
    }} else {{
        int v_row = out_row - N_Q - N_GATE - N_K + (int)slid;
{v_write}
    }}
"""


_batched_proj_cache = {}


def _get_batched_proj_kernel(K, N_Q, N_GATE, N_K, N_V, B, group_size=64):
    key = (K, N_Q, N_GATE, N_K, N_V, B, group_size)
    if key not in _batched_proj_cache:
        _batched_proj_cache[key] = mx.fast.metal_kernel(
            name=f"batched_fused_gqa_proj_K{K}_NQ{N_Q}_B{B}",
            input_names=["x", "W_merged", "S_merged", "B_merged"],
            output_names=["q_out", "gate_out", "k_out", "v_out"],
            source=_gen_batched_fused_gqa_proj_source(K, N_Q, N_GATE, N_K, N_V, B, group_size),
        )
    return _batched_proj_cache[key]


def batched_fused_gqa_projections(x, W_merged, S_merged, B_merged, proj_dims,
                                   batch_size, total_tg=None):
    """Batched fused GQA projections with register weight sharing.

    Args:
        x: [B, 1, K] bf16
        W_merged, S_merged, B_merged: merged q+gate+k+v weights
        proj_dims: (N_Q, N_GATE, N_K, N_V)
        batch_size: B (1..8)

    Returns:
        queries (B, 1, N_Q) bf16, gate_sigmoid (B, 1, N_GATE) f32,
        keys (B, 1, N_K) bf16, values (B, 1, N_V) bf16
    """
    B = batch_size
    N_Q, N_GATE, N_K, N_V = proj_dims
    K = x.shape[-1]

    kern = _get_batched_proj_kernel(K, N_Q, N_GATE, N_K, N_V, B)

    if total_tg is None:
        total_tg = ceil_div(N_Q, 8) + ceil_div(N_GATE, 8) + ceil_div(N_K, 8) + ceil_div(N_V, 8)

    x_flat = x.reshape(B, K)

    results = kern(
        inputs=[x_flat, W_merged, S_merged, B_merged],
        output_shapes=[
            (B * N_Q,), (B * N_GATE,), (B * N_K,), (B * N_V,),
        ],
        output_dtypes=[mx.bfloat16, mx.float32, mx.bfloat16, mx.bfloat16],
        grid=(32, total_tg * 2, 1),
        threadgroup=(32, 2, 1),
    )

    return (results[0].reshape(B, 1, N_Q),
            results[1].reshape(B, 1, N_GATE),
            results[2].reshape(B, 1, N_K),
            results[3].reshape(B, 1, N_V))
