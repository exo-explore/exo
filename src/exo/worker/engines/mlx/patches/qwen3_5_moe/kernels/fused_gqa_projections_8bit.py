"""Fused GQA projections for Qwen3.5 (Dispatch 1).

Single dispatch fuses 4 quantized 8-bit GEMVs with region-specific epilogues:
  - q_proj queries: GEMV → raw bf16 write
  - q_proj gate:    GEMV → sigmoid → f32 write
  - k_proj:         GEMV → raw bf16 write
  - v_proj:         GEMV → raw bf16 write

All constants baked into Metal source (no scalar kernel inputs).
"""

import mlx.core as mx


def ceil_div(a, b):
    return (a + b - 1) // b


def _gen_fused_gqa_projections_source(K, N_Q, N_GATE, N_K, N_V, group_size=64):
    gs = int(group_size)
    sc_stride = 256 // gs
    slid_div = gs // 8
    N_TOTAL = N_Q + N_GATE + N_K + N_V
    K_groups = K // gs
    N_Q_TG = ceil_div(N_Q, 8)
    N_GATE_TG = ceil_div(N_GATE, 8)
    N_K_TG = ceil_div(N_K, 8)

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
    const int N_TOTAL = {N_TOTAL};
    const int N_Q_TG = {N_Q_TG};
    const int N_GATE_TG = {N_GATE_TG};
    const int N_K_TG = {N_K_TG};

    uint3 tgid = threadgroup_position_in_grid;
    uint sgid = simdgroup_index_in_threadgroup;
    uint slid = thread_index_in_simdgroup;
    int b_idx = tgid.z;
    int tg = tgid.y;

    int out_row;
    int region;

    if (tg < N_Q_TG) {{
        region = 0;
        out_row = tg * 8 + sgid * RESULTS_PER_SG;
    }} else if (tg < N_Q_TG + N_GATE_TG) {{
        region = 1;
        out_row = N_Q + (tg - N_Q_TG) * 8 + sgid * RESULTS_PER_SG;
    }} else if (tg < N_Q_TG + N_GATE_TG + N_K_TG) {{
        region = 2;
        out_row = N_Q + N_GATE + (tg - N_Q_TG - N_GATE_TG) * 8 + sgid * RESULTS_PER_SG;
    }} else {{
        region = 3;
        out_row = N_Q + N_GATE + N_K + (tg - N_Q_TG - N_GATE_TG - N_K_TG) * 8 + sgid * RESULTS_PER_SG;
    }}

    if (out_row >= N_TOTAL) return;

    const device uint8_t* ws = (const device uint8_t*)W_merged + (long)out_row * K + slid * VALUES_PER_THREAD;
    const device bfloat16_t* sc = (const device bfloat16_t*)S_merged + (long)out_row * K_groups + slid / SLID_DIV;
    const device bfloat16_t* bi = (const device bfloat16_t*)B_merged + (long)out_row * K_groups + slid / SLID_DIV;

    float result[4] = {{0, 0, 0, 0}};
    int x_base = b_idx * K + slid * VALUES_PER_THREAD;

    for (int k_off = 0; k_off < K; k_off += BLOCK_SIZE) {{
        float x_thread[8];
        float xsum = 0;
        for (int i = 0; i < 8; i++) {{
            float xi = float(x[x_base + i]);
            x_thread[i] = xi;
            xsum += xi;
        }}

        for (int row = 0; row < RESULTS_PER_SG; row++) {{
            const device uint8_t* w = ws + row * K;
            float s_val = float(sc[row * K_groups]);
            float b_val = float(bi[row * K_groups]);
            float accum = 0;
            for (int i = 0; i < 8; i++) {{
                accum += x_thread[i] * float(w[i]);
            }}
            result[row] += s_val * accum + xsum * b_val;
        }}

        ws += BLOCK_SIZE;
        sc += SC_STRIDE;
        bi += SC_STRIDE;
        x_base += BLOCK_SIZE;
    }}

    for (int row = 0; row < RESULTS_PER_SG; row++) {{
        result[row] = simd_sum(result[row]);
    }}

    if (region == 0) {{
        int q_row = out_row + (int)slid;
        if (slid < (uint)RESULTS_PER_SG && q_row < N_Q) {{
            q_out[b_idx * N_Q + q_row] = static_cast<bfloat16_t>(result[slid]);
        }}
    }} else if (region == 1) {{
        int g_row = out_row - N_Q + (int)slid;
        if (slid < (uint)RESULTS_PER_SG && g_row < N_GATE) {{
            float val = result[slid];
            float sig = 1.0f / (1.0f + metal::exp(-val));
            gate_out[b_idx * N_GATE + g_row] = sig;
        }}
    }} else if (region == 2) {{
        int k_row = out_row - N_Q - N_GATE + (int)slid;
        if (slid < (uint)RESULTS_PER_SG && k_row < N_K) {{
            k_out[b_idx * N_K + k_row] = static_cast<bfloat16_t>(result[slid]);
        }}
    }} else {{
        int v_row = out_row - N_Q - N_GATE - N_K + (int)slid;
        int N_V = N_TOTAL - N_Q - N_GATE - N_K;
        if (slid < (uint)RESULTS_PER_SG && v_row < N_V) {{
            v_out[b_idx * N_V + v_row] = static_cast<bfloat16_t>(result[slid]);
        }}
    }}
"""


_fused_gqa_proj_cache = {}


def _get_fused_gqa_proj_kernel(K, N_Q, N_GATE, N_K, N_V, group_size=64):
    key = (K, N_Q, N_GATE, N_K, N_V, group_size)
    if key not in _fused_gqa_proj_cache:
        _fused_gqa_proj_cache[key] = mx.fast.metal_kernel(
            name=f"fused_gqa_proj_K{K}_NQ{N_Q}_NG{N_GATE}_NK{N_K}_NV{N_V}",
            input_names=["x", "W_merged", "S_merged", "B_merged"],
            output_names=["q_out", "gate_out", "k_out", "v_out"],
            source=_gen_fused_gqa_projections_source(K, N_Q, N_GATE, N_K, N_V, group_size),
        )
    return _fused_gqa_proj_cache[key]


def fused_gqa_projections(
    x, W_merged, S_merged, B_merged, proj_dims, batch_size=1,
    scalars=None, total_tg=None,
):
    B = batch_size
    N_Q, N_GATE, N_K, N_V = proj_dims
    K = x.shape[-1]

    kern = _get_fused_gqa_proj_kernel(K, N_Q, N_GATE, N_K, N_V)

    if total_tg is None:
        total_tg = ceil_div(N_Q, 8) + ceil_div(N_GATE, 8) + ceil_div(N_K, 8) + ceil_div(N_V, 8)

    x_flat = x.reshape(B, K)

    results = kern(
        inputs=[x_flat, W_merged, S_merged, B_merged],
        output_shapes=[
            (B * N_Q,), (B * N_GATE,), (B * N_K,), (B * N_V,),
        ],
        output_dtypes=[mx.bfloat16, mx.float32, mx.bfloat16, mx.bfloat16],
        grid=(32, total_tg * 2, B),
        threadgroup=(32, 2, 1),
    )

    return (results[0].reshape(B, 1, N_Q),
            results[1].reshape(B, 1, N_GATE),
            results[2].reshape(B, 1, N_K),
            results[3].reshape(B, 1, N_V))
