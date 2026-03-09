"""Dispatch 1: Fused o_proj (8-bit) + gate GEMV parts + x² partials for Qwen3.5.

Port of Kimi's custom_oproj_gate_gemv.py adapted for 8-bit quantized o_proj.

Single dispatch with 3 GEMV types sharing 256-thread TGs (8 SGs of 32):

TGs 0 to N_OPROJ_TG-1: o_proj GEMV (8-bit, M=4096, K=8192)
  8-bit affine dequant: result = scale * Σ(x*w_uint8) + bias * Σ(x)
  Epilogue: h = oproj_result + residual, h_scaled = h*w_rms, h_out = h, x²_acc
  TG reduction: 8 SG x² → 1 float per TG.

TGs N_OPROJ_TG to +N_M1_TG-1: M1 GEMV (bf16, E × K_attn)
  M1 = W_fused @ W_oproj (pre-computed). Input = attn_out.
  Output: gate_part_a (E,) f32.

TGs +N_M1_TG to end: W_fused GEMV (bf16, E × K)
  W_fused × residual → gate_part_b (E,) f32.
"""
import mlx.core as mx


def ceil_div(a, b):
    return (a + b - 1) // b


def _gen_custom_oproj_8bit_source(n_experts=64, group_size=64, scale_bf16=True):
    """Generate Metal source for fused 8-bit o_proj + bf16 gate GEMVs."""
    E = int(n_experts)
    gs = int(group_size)
    sc_t = "bfloat16_t" if scale_bf16 else "float"
    # 8-bit dequant params for o_proj
    oproj_slid_divisor = gs // 8     # 64/8 = 8
    oproj_sc_stride = 256 // gs      # 256/64 = 4

    return f"""
    // ── Constants ──
    const int TM = 4;       // rows per SG for bf16 GEMVs
    const int TN = 4;       // elements per thread per iter for bf16 GEMVs
    const int blockN = 128; // 32 threads × TN=4
    const int E_CONST = {E};

    int M = M_val;                   // 4096 (hidden_size)
    int K_attn = K_attn_val;         // 8192 (o_proj input dim)
    int K_hidden = K_hidden_val;     // 4096 (hidden_size, same as M for Qwen)
    int N_OPROJ_TG = N_OPROJ_TG_val;
    int N_M1_TG = N_M1_TG_val;
    int blockM_gate = BM_GATE_val * TM;  // rows per gate GEMV TG

    uint tg_x = threadgroup_position_in_grid.x;
    uint sgid = simdgroup_index_in_threadgroup;  // 0..7
    uint slid = thread_index_in_simdgroup;       // 0..31

    if (tg_x < (uint)N_OPROJ_TG) {{
        // ════════════════════════════════════════════════════════════════
        // O_PROJ GEMV: 8-bit quantized, M=4096, K_attn=8192
        // Each TG handles 32 rows (8 SGs × 4 rows each)
        // 8-bit affine dequant: result = scale * Σ(x*w) + bias * Σ(x)
        // ════════════════════════════════════════════════════════════════
        const int blockM = 32;  // 8 SGs × TM=4
        const int VPT = 8;     // values per thread per iteration
        const int BLOCK_SIZE = 256;  // 32 threads × 8 values

        int out_row = int(tg_x) * blockM + int(sgid) * TM;
        if (out_row >= M) return;
        out_row = (out_row + TM <= M) ? out_row : (M - TM);

        threadgroup float tgp_x2[8];

        // 8-bit GEMV K-loop: K-outer, tm-inner (input loaded once, reused across TM rows)
        float acc[TM] = {{0.0f, 0.0f, 0.0f, 0.0f}};
        float result[TM];
        int K_groups = K_attn / {gs};

        // Per-row weight/scale/bias pointers
        const device uint8_t* ws0 = (const device uint8_t*)W_oproj + (long)(out_row + 0) * K_attn + slid * VPT;
        const device uint8_t* ws1 = (const device uint8_t*)W_oproj + (long)(out_row + 1) * K_attn + slid * VPT;
        const device uint8_t* ws2 = (const device uint8_t*)W_oproj + (long)(out_row + 2) * K_attn + slid * VPT;
        const device uint8_t* ws3 = (const device uint8_t*)W_oproj + (long)(out_row + 3) * K_attn + slid * VPT;

        const device {sc_t}* sc0 = (const device {sc_t}*)S_oproj + (long)(out_row + 0) * K_groups + slid / {oproj_slid_divisor};
        const device {sc_t}* sc1 = (const device {sc_t}*)S_oproj + (long)(out_row + 1) * K_groups + slid / {oproj_slid_divisor};
        const device {sc_t}* sc2 = (const device {sc_t}*)S_oproj + (long)(out_row + 2) * K_groups + slid / {oproj_slid_divisor};
        const device {sc_t}* sc3 = (const device {sc_t}*)S_oproj + (long)(out_row + 3) * K_groups + slid / {oproj_slid_divisor};

        const device {sc_t}* bi0 = (const device {sc_t}*)B_oproj + (long)(out_row + 0) * K_groups + slid / {oproj_slid_divisor};
        const device {sc_t}* bi1 = (const device {sc_t}*)B_oproj + (long)(out_row + 1) * K_groups + slid / {oproj_slid_divisor};
        const device {sc_t}* bi2 = (const device {sc_t}*)B_oproj + (long)(out_row + 2) * K_groups + slid / {oproj_slid_divisor};
        const device {sc_t}* bi3 = (const device {sc_t}*)B_oproj + (long)(out_row + 3) * K_groups + slid / {oproj_slid_divisor};

        int xb = slid * VPT;

        for (int k = 0; k < K_attn; k += BLOCK_SIZE) {{
            // Load input ONCE per K-block
            float xv[VPT];
            float xsum = 0.0f;
            for (int i = 0; i < VPT; i++) {{
                xv[i] = float(attn_out[xb + i]);
                xsum += xv[i];
            }}

            // Row 0
            {{ float wacc = 0.0f;
              for (int i = 0; i < VPT; i++) wacc += xv[i] * float(ws0[i]);
              acc[0] += float(*sc0) * wacc + xsum * float(*bi0);
              ws0 += BLOCK_SIZE; sc0 += {oproj_sc_stride}; bi0 += {oproj_sc_stride}; }}
            // Row 1
            {{ float wacc = 0.0f;
              for (int i = 0; i < VPT; i++) wacc += xv[i] * float(ws1[i]);
              acc[1] += float(*sc1) * wacc + xsum * float(*bi1);
              ws1 += BLOCK_SIZE; sc1 += {oproj_sc_stride}; bi1 += {oproj_sc_stride}; }}
            // Row 2
            {{ float wacc = 0.0f;
              for (int i = 0; i < VPT; i++) wacc += xv[i] * float(ws2[i]);
              acc[2] += float(*sc2) * wacc + xsum * float(*bi2);
              ws2 += BLOCK_SIZE; sc2 += {oproj_sc_stride}; bi2 += {oproj_sc_stride}; }}
            // Row 3
            {{ float wacc = 0.0f;
              for (int i = 0; i < VPT; i++) wacc += xv[i] * float(ws3[i]);
              acc[3] += float(*sc3) * wacc + xsum * float(*bi3);
              ws3 += BLOCK_SIZE; sc3 += {oproj_sc_stride}; bi3 += {oproj_sc_stride}; }}

            xb += BLOCK_SIZE;
        }}

        for (int tm = 0; tm < TM; tm++)
            result[tm] = simd_sum(acc[tm]);

        // Epilogue: addmm + x² + h_scaled + h_out
        float x2_acc = 0.0f;
        if (slid == 0) {{
            for (int tm = 0; tm < TM; tm++) {{
                int k = out_row + tm;
                float h = result[tm] + float(residual[k]);
                x2_acc += h * h;
                h_scaled[k] = static_cast<bfloat16_t>(h * float(w_rms[k]));
                h_out[k] = static_cast<bfloat16_t>(h);
            }}
        }}

        // TG x² reduction: 8 SGs → 1 value
        if (slid == 0) tgp_x2[sgid] = x2_acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0 && slid == 0) {{
            float total = 0.0f;
            for (int s = 0; s < 8; s++) total += tgp_x2[s];
            x2_partials[tg_x] = total;
        }}

    }} else if (tg_x < (uint)(N_OPROJ_TG + N_M1_TG)) {{
        // ════════════════════════════════════════════════════════════════
        // M1 GEMV: bf16, M=E, K=K_attn — M1 × attn_out → gate_part_a
        // ════════════════════════════════════════════════════════════════
        int local_tg = int(tg_x) - N_OPROJ_TG;
        int out_row = local_tg * blockM_gate + int(sgid) * TM;
        if (out_row >= E_CONST) return;
        out_row = (out_row + TM <= E_CONST) ? out_row : (E_CONST - TM);

        float result[TM] = {{0.0f, 0.0f, 0.0f, 0.0f}};
        int bn = int(slid) * TN;
        int n_iter = K_attn / blockN;

        for (int i = 0; i < n_iter; i++) {{
            float v[TN];
            for (int tn = 0; tn < TN; tn++)
                v[tn] = float(attn_out[bn + tn]);
            for (int tm = 0; tm < TM; tm++) {{
                float acc = 0.0f;
                for (int tn = 0; tn < TN; tn++)
                    acc += float(M1[(out_row + tm) * K_attn + bn + tn]) * v[tn];
                result[tm] += acc;
            }}
            bn += blockN;
        }}

        // Handle remainder if K_attn not divisible by blockN
        if (K_attn > n_iter * blockN) {{
            for (int tm = 0; tm < TM; tm++) {{
                for (int tn = 0; tn < TN; tn++) {{
                    if (bn + tn < K_attn)
                        result[tm] += float(M1[(out_row + tm) * K_attn + bn + tn])
                                    * float(attn_out[bn + tn]);
                }}
            }}
        }}

        for (int tm = 0; tm < TM; tm++)
            result[tm] = simd_sum(result[tm]);

        if (slid == 0) {{
            for (int tm = 0; tm < TM; tm++) {{
                int e = out_row + tm;
                if (e < E_CONST)
                    gate_part_a[e] = result[tm];
            }}
        }}

    }} else {{
        // ════════════════════════════════════════════════════════════════
        // W_FUSED GEMV: bf16, M=E, K=K_hidden — W_fused × residual → gate_part_b
        // ════════════════════════════════════════════════════════════════
        int local_tg = int(tg_x) - N_OPROJ_TG - N_M1_TG;
        int out_row = local_tg * blockM_gate + int(sgid) * TM;
        if (out_row >= E_CONST) return;
        out_row = (out_row + TM <= E_CONST) ? out_row : (E_CONST - TM);

        float result[TM] = {{0.0f, 0.0f, 0.0f, 0.0f}};
        int bn = int(slid) * TN;
        int n_iter = K_hidden / blockN;

        for (int i = 0; i < n_iter; i++) {{
            float v[TN];
            for (int tn = 0; tn < TN; tn++)
                v[tn] = float(residual[bn + tn]);
            for (int tm = 0; tm < TM; tm++) {{
                float acc = 0.0f;
                for (int tn = 0; tn < TN; tn++)
                    acc += float(W_fused[(out_row + tm) * K_hidden + bn + tn]) * v[tn];
                result[tm] += acc;
            }}
            bn += blockN;
        }}

        if (K_hidden > n_iter * blockN) {{
            for (int tm = 0; tm < TM; tm++) {{
                for (int tn = 0; tn < TN; tn++) {{
                    if (bn + tn < K_hidden)
                        result[tm] += float(W_fused[(out_row + tm) * K_hidden + bn + tn])
                                    * float(residual[bn + tn]);
                }}
            }}
        }}

        for (int tm = 0; tm < TM; tm++)
            result[tm] = simd_sum(result[tm]);

        if (slid == 0) {{
            for (int tm = 0; tm < TM; tm++) {{
                int e = out_row + tm;
                if (e < E_CONST)
                    gate_part_b[e] = result[tm];
            }}
        }}
    }}
"""


_custom_oproj_8bit_kernels = {}


def _get_custom_oproj_8bit_kernel(n_experts=64, group_size=64, scale_bf16=True):
    key = (n_experts, group_size, scale_bf16)
    if key not in _custom_oproj_8bit_kernels:
        sc_tag = "_bf16sc" if scale_bf16 else ""
        _custom_oproj_8bit_kernels[key] = mx.fast.metal_kernel(
            name=f"custom_oproj_gate_gemv_8bit_e{n_experts}_gs{group_size}{sc_tag}",
            input_names=[
                "W_oproj", "S_oproj", "B_oproj",   # o_proj 8-bit weights
                "attn_out",                          # (K_attn,) bf16
                "residual",                          # (K,) bf16
                "w_rms",                             # (K,) bf16 — RMSNorm weight
                "M1",                                # (E, K_attn) bf16
                "W_fused",                           # (E, K) bf16
                "M_val", "K_attn_val", "K_hidden_val",
                "N_OPROJ_TG_val", "N_M1_TG_val", "BM_GATE_val",
            ],
            output_names=["h_scaled", "h_out", "x2_partials",
                          "gate_part_a", "gate_part_b"],
            source=_gen_custom_oproj_8bit_source(n_experts, group_size, scale_bf16),
        )
    return _custom_oproj_8bit_kernels[key]


def fused_custom_oproj_8bit(W_oproj, S_oproj, B_oproj,
                             attn_out, residual, w_rms,
                             M1, W_fused,
                             M, K_attn, K_hidden=None,
                             n_experts=64, gate_bm=8,
                             group_size=64):
    """Dispatch the fused 8-bit o_proj + bf16 gate GEMVs kernel.

    Three GEMV types in one dispatch:
    - TGs 0..N_OPROJ_TG-1: 8-bit o_proj GEMV → h_scaled, h_out, x2_partials
    - TGs N_OPROJ_TG..+N_M1_TG: bf16 M1 GEMV → gate_part_a
    - TGs +N_M1_TG..end: bf16 W_fused GEMV → gate_part_b

    Args:
        W_oproj/S_oproj/B_oproj: 8-bit quantized o_proj weights
        attn_out: (K_attn,) bf16 — pre-o_proj attention output
        residual: (K,) bf16 — input residual
        w_rms: (K,) bf16 — post_attention_layernorm weight
        M1: (E, K_attn) bf16 — precomputed W_fused @ W_oproj
        W_fused: (E, K) bf16 — precomputed dequant(W_gate) * w_rms
        M: hidden size (4096)
        K_attn: attention output dim (8192)
        K_hidden: hidden size for W_fused (defaults to M)
        gate_bm: SGs per gate TG (1,2,4,8). Controls TG count.

    Returns:
        (h_scaled, h_out, x2_partials, gate_part_a, gate_part_b)
    """
    M = int(M)
    K_attn = int(K_attn)
    K_hidden = int(K_hidden) if K_hidden is not None else M
    scale_bf16 = (S_oproj.dtype == mx.bfloat16)

    kern = _get_custom_oproj_8bit_kernel(n_experts, group_size, scale_bf16)

    n_oproj_tg = ceil_div(M, 32)  # 128 for M=4096
    blockM_gate = gate_bm * 4     # rows per gate GEMV TG
    n_m1_tg = ceil_div(n_experts, blockM_gate)
    n_wf_tg = ceil_div(n_experts, blockM_gate)
    total_tg = n_oproj_tg + n_m1_tg + n_wf_tg

    results = kern(
        inputs=[W_oproj, S_oproj, B_oproj,
                attn_out, residual, w_rms, M1, W_fused,
                M, K_attn, K_hidden, n_oproj_tg, n_m1_tg, gate_bm],
        output_shapes=[(M,), (M,), (n_oproj_tg,), (n_experts,), (n_experts,)],
        output_dtypes=[mx.bfloat16, mx.bfloat16, mx.float32,
                       mx.float32, mx.float32],
        grid=(total_tg * 32, 8, 1),    # total threads: total_tg * 256
        threadgroup=(32, 8, 1),          # 256 threads per TG (8 SGs of 32)
    )
    return results[0], results[1], results[2], results[3], results[4]
