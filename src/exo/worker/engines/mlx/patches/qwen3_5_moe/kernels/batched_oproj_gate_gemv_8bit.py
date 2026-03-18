"""Batched Dispatch 1: Fused o_proj (8-bit) + gate GEMV parts + x² partials.

Adapts custom_oproj_gate_gemv_8bit for batch_size B (1..8).
Register-level weight sharing: each TG loads weights once, computes B outputs.

Three GEMV regions (same as B=1):
  TGs 0..N_OPROJ_TG-1: o_proj GEMV (8-bit) + residual + h_scaled + x²
  TGs N_OPROJ_TG..+N_M1_TG-1: M1 × attn_out → gate_part_a (bf16 GEMV)
  TGs +N_M1_TG..end: W_fused × residual → gate_part_b (bf16 GEMV)

All constants baked into Metal source. B is unrolled at code-generation time.
"""
import mlx.core as mx


def ceil_div(a, b):
    return (a + b - 1) // b


def _gen_batched_oproj_source(n_experts, M, K_attn, K_hidden, B, group_size=64, gate_bm=8):
    E = int(n_experts)
    gs = group_size
    oproj_slid_divisor = gs // 8
    oproj_sc_stride = 256 // gs
    blockM_gate = gate_bm * 4
    n_m1_tg = ceil_div(E, blockM_gate)

    # Generate unrolled per-batch code for o_proj epilogue
    oproj_epilogue = []
    for b in range(B):
        oproj_epilogue.append(f"""
            float x2_acc{b} = 0.0f;
            for (int tm = 0; tm < TM; tm++) {{
                int k = out_row + tm;
                float h{b} = result{b}[tm] + float(residual[{b} * M_DIM + k]);
                x2_acc{b} += h{b} * h{b};
                h_scaled[{b} * M_DIM + k] = static_cast<bfloat16_t>(h{b} * float(w_rms[k]));
                h_out[{b} * M_DIM + k] = static_cast<bfloat16_t>(h{b});
            }}""")
    oproj_epilogue_code = "\n".join(oproj_epilogue)

    oproj_x2_write = []
    for b in range(B):
        oproj_x2_write.append(f"""
                total{b} += tgp_x2[s * {B} + {b}];""")
    oproj_x2_sum = "\n".join(oproj_x2_write)

    oproj_x2_final = []
    for b in range(B):
        oproj_x2_final.append(f"""
                x2_partials[{b} * N_OPROJ_TG_DIM + tg_x] = total{b};""")
    oproj_x2_final_code = "\n".join(oproj_x2_final)

    # o_proj K-loop: load weights once, compute B batch elements
    oproj_x_load = "\n".join(f"""
            float xv{b}[VPT]; float xsum{b} = 0.0f;
            for (int i = 0; i < VPT; i++) {{ xv{b}[i] = float(attn_out[{b} * K_ATTN_DIM + xb + i]); xsum{b} += xv{b}[i]; }}""" for b in range(B))

    oproj_qdot = "\n".join(f"""
                acc{b}[row] += s_val * wdot(xv{b}, w_vals) + xsum{b} * b_val;""" for b in range(B))

    oproj_result_decls = " ".join(f"float acc{b}[TM] = {{0,0,0,0}};" for b in range(B))
    oproj_simd_reduce = "\n".join(f"            float result{b}[TM]; for (int tm=0;tm<TM;tm++) result{b}[tm] = simd_sum(acc{b}[tm]);" for b in range(B))

    oproj_tgp_write = "\n".join(f"            tgp_x2[sgid * {B} + {b}] = x2_acc{b};" for b in range(B))
    oproj_total_decls = " ".join(f"float total{b} = 0.0f;" for b in range(B))

    # Gate M1 GEMV: load M1 weights once, compute B dot products with B attn_outs
    gate_a_x_load = "\n".join(f"""
            float v{b}[TN];
            for (int tn = 0; tn < TN; tn++) v{b}[tn] = float(attn_out[{b} * K_ATTN_DIM + bn + tn]);""" for b in range(B))

    gate_a_dot = "\n".join(f"""
                float gacc{b} = 0.0f;
                for (int tn = 0; tn < TN; tn++) gacc{b} += w_row[tn] * v{b}[tn];
                gresult{b}[tm] += gacc{b};""" for b in range(B))

    gate_a_decls = " ".join(f"float gresult{b}[TM] = {{0,0,0,0}};" for b in range(B))
    gate_a_reduce = "\n".join(f"            gresult{b}[tm] = simd_sum(gresult{b}[tm]);" for b in range(B))
    gate_a_write = "\n".join(f"""
                    gate_part_a[{b} * E_CONST + e] = gresult{b}[tm];""" for b in range(B))

    # Gate W_fused GEMV: same pattern but with residual input
    gate_b_x_load = "\n".join(f"""
            float rv{b}[TN];
            for (int tn = 0; tn < TN; tn++) rv{b}[tn] = float(residual[{b} * K_HIDDEN_DIM + bn + tn]);""" for b in range(B))

    gate_b_dot = "\n".join(f"""
                float wdot{b} = 0.0f;
                for (int tn = 0; tn < TN; tn++) wdot{b} += w_row[tn] * rv{b}[tn];
                bresult{b}[tm] += wdot{b};""" for b in range(B))

    gate_b_decls = " ".join(f"float bresult{b}[TM] = {{0,0,0,0}};" for b in range(B))
    gate_b_reduce = "\n".join(f"            bresult{b}[tm] = simd_sum(bresult{b}[tm]);" for b in range(B))
    gate_b_write = "\n".join(f"""
                    gate_part_b[{b} * E_CONST + e] = bresult{b}[tm];""" for b in range(B))

    return f"""
    const int TM = 4;
    const int TN = 4;
    const int blockN = 128;
    const int E_CONST = {E};
    const int M_DIM = {M};
    const int K_ATTN_DIM = {K_attn};
    const int K_HIDDEN_DIM = {K_hidden};
    const int N_OPROJ_TG_DIM = {ceil_div(M, 32)};
    const int BATCH_SIZE = {B};

    // Helper: dot product of x_thread and w_vals (8 elements)
    auto wdot = [](thread float* x, thread float* w) -> float {{
        float a = 0;
        for (int i = 0; i < 8; i++) a += x[i] * w[i];
        return a;
    }};

    const int N_OPROJ_TG = N_OPROJ_TG_DIM;
    const int N_M1_TG = {n_m1_tg};
    const int blockM_gate = {blockM_gate};

    uint tg_x = threadgroup_position_in_grid.x;
    uint sgid = simdgroup_index_in_threadgroup;
    uint slid = thread_index_in_simdgroup;

    if (tg_x < (uint)N_OPROJ_TG) {{
        // ══════ O_PROJ GEMV (8-bit, register-sharing for {B} batches) ══════
        const int blockM = 32;
        const int VPT = 8;
        const int BLOCK_SIZE = 256;

        int out_row = int(tg_x) * blockM + int(sgid) * TM;
        if (out_row >= M_DIM) return;
        out_row = (out_row + TM <= M_DIM) ? out_row : (M_DIM - TM);

        threadgroup float tgp_x2[8 * {B}];

        {oproj_result_decls}
        int K_groups = K_ATTN_DIM / {gs};

        // Weight pointers (shared across batch)
        const device uint8_t* ws = (const device uint8_t*)W_oproj
            + (long)out_row * K_ATTN_DIM + slid * VPT;
        const device bfloat16_t* sc = (const device bfloat16_t*)S_oproj
            + (long)out_row * K_groups + slid / {oproj_slid_divisor};
        const device bfloat16_t* bi = (const device bfloat16_t*)B_oproj
            + (long)out_row * K_groups + slid / {oproj_slid_divisor};

        int xb = slid * VPT;

        for (int k = 0; k < K_ATTN_DIM; k += BLOCK_SIZE) {{
            // Load x for all {B} batches
{oproj_x_load}

            // Load weights once, compute all batches
            for (int row = 0; row < TM; row++) {{
                const device uint8_t* wl = ws + row * K_ATTN_DIM;
                float s_val = float(sc[row * K_groups]);
                float b_val = float(bi[row * K_groups]);
                float w_vals[VPT];
                for (int i = 0; i < VPT; i++) w_vals[i] = float(wl[i]);
{oproj_qdot}
            }}

            ws += BLOCK_SIZE; sc += {oproj_sc_stride}; bi += {oproj_sc_stride};
            xb += BLOCK_SIZE;
        }}

        // simd_sum for all batches
{oproj_simd_reduce}

        // Epilogue: residual add + x² + h_scaled + h_out
        if (slid == 0) {{
{oproj_epilogue_code}
{oproj_tgp_write}
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0 && slid == 0) {{
            {oproj_total_decls}
            for (int s = 0; s < 8; s++) {{
{oproj_x2_sum}
            }}
{oproj_x2_final_code}
        }}

    }} else if (tg_x < (uint)(N_OPROJ_TG + N_M1_TG)) {{
        // ══════ M1 GEMV (bf16, register-sharing for {B} batches) ══════
        int local_tg = int(tg_x) - N_OPROJ_TG;
        int out_row = local_tg * blockM_gate + int(sgid) * TM;
        if (out_row >= E_CONST) return;
        out_row = (out_row + TM <= E_CONST) ? out_row : (E_CONST - TM);

        {gate_a_decls}
        int bn = int(slid) * TN;
        int n_iter = K_ATTN_DIM / blockN;

        for (int i = 0; i < n_iter; i++) {{
{gate_a_x_load}
            for (int tm = 0; tm < TM; tm++) {{
                float w_row[TN];
                for (int tn = 0; tn < TN; tn++) w_row[tn] = float(M1[(out_row + tm) * K_ATTN_DIM + bn + tn]);
{gate_a_dot}
            }}
            bn += blockN;
        }}

        for (int tm = 0; tm < TM; tm++) {{
{gate_a_reduce}
        }}

        if (slid == 0) {{
            for (int tm = 0; tm < TM; tm++) {{
                int e = out_row + tm;
                if (e < E_CONST) {{
{gate_a_write}
                }}
            }}
        }}

    }} else {{
        // ══════ W_FUSED GEMV (bf16, register-sharing for {B} batches) ══════
        int local_tg = int(tg_x) - N_OPROJ_TG - N_M1_TG;
        int out_row = local_tg * blockM_gate + int(sgid) * TM;
        if (out_row >= E_CONST) return;
        out_row = (out_row + TM <= E_CONST) ? out_row : (E_CONST - TM);

        {gate_b_decls}
        int bn = int(slid) * TN;
        int n_iter = K_HIDDEN_DIM / blockN;

        for (int i = 0; i < n_iter; i++) {{
{gate_b_x_load}
            for (int tm = 0; tm < TM; tm++) {{
                float w_row[TN];
                for (int tn = 0; tn < TN; tn++) w_row[tn] = float(W_fused[(out_row + tm) * K_HIDDEN_DIM + bn + tn]);
{gate_b_dot}
            }}
            bn += blockN;
        }}

        for (int tm = 0; tm < TM; tm++) {{
{gate_b_reduce}
        }}

        if (slid == 0) {{
            for (int tm = 0; tm < TM; tm++) {{
                int e = out_row + tm;
                if (e < E_CONST) {{
{gate_b_write}
                }}
            }}
        }}
    }}
"""


_batched_oproj_cache = {}


def _get_batched_oproj_kernel(n_experts, M, K_attn, K_hidden, B, group_size=64, gate_bm=8):
    key = (n_experts, M, K_attn, K_hidden, B, group_size, gate_bm)
    if key not in _batched_oproj_cache:
        _batched_oproj_cache[key] = mx.fast.metal_kernel(
            name=f"batched_oproj_E{n_experts}_M{M}_Ka{K_attn}_Kh{K_hidden}_B{B}",
            input_names=[
                "W_oproj", "S_oproj", "B_oproj",
                "attn_out", "residual", "w_rms",
                "M1", "W_fused",
            ],
            output_names=["h_scaled", "h_out", "x2_partials",
                          "gate_part_a", "gate_part_b"],
            source=_gen_batched_oproj_source(n_experts, M, K_attn, K_hidden, B, group_size, gate_bm),
        )
    return _batched_oproj_cache[key]


def batched_oproj_gate_gemv(W_oproj, S_oproj, B_oproj,
                             attn_out, residual, w_rms,
                             M1, W_fused,
                             M, K_attn, batch_size,
                             n_experts=256, gate_bm=8,
                             K_hidden=None, group_size=64):
    """Batched fused 8-bit o_proj + bf16 gate GEMVs.

    Args:
        W_oproj/S_oproj/B_oproj: 8-bit o_proj weights
        attn_out: (B, K_attn) bf16
        residual: (B, K) bf16
        w_rms: (K,) bf16 — RMSNorm weight (shared)
        M1: (E, K_attn) bf16 (shared)
        W_fused: (E, K) bf16 (shared)
        M: hidden size
        K_attn: attention output dim
        batch_size: B
        n_experts: E
        gate_bm: SGs per gate TG

    Returns:
        h_scaled (B, M) bf16, h_out (B, M) bf16,
        x2_partials (B, N_TG) f32, gate_part_a (B, E) f32, gate_part_b (B, E) f32
    """
    B = batch_size
    M_val = int(M)
    K_attn_val = int(K_attn)
    K_hidden_val = int(K_hidden) if K_hidden is not None else M_val

    kern = _get_batched_oproj_kernel(n_experts, M_val, K_attn_val, K_hidden_val, B, group_size, gate_bm)

    n_oproj_tg = ceil_div(M_val, 32)
    blockM_gate = gate_bm * 4
    n_m1_tg = ceil_div(n_experts, blockM_gate)
    n_wf_tg = ceil_div(n_experts, blockM_gate)
    total_tg = n_oproj_tg + n_m1_tg + n_wf_tg

    results = kern(
        inputs=[W_oproj, S_oproj, B_oproj,
                attn_out.reshape(B * K_attn_val), residual.reshape(B * M_val), w_rms,
                M1, W_fused],
        output_shapes=[
            (B * M_val,), (B * M_val,),
            (B * n_oproj_tg,),
            (B * n_experts,), (B * n_experts,),
        ],
        output_dtypes=[mx.bfloat16, mx.bfloat16, mx.float32, mx.float32, mx.float32],
        grid=(total_tg * 32, 8, 1),
        threadgroup=(32, 8, 1),
    )

    return (results[0].reshape(B, M_val),
            results[1].reshape(B, M_val),
            results[2].reshape(B, n_oproj_tg),
            results[3].reshape(B, n_experts),
            results[4].reshape(B, n_experts))
