"""Dispatch 2: SwiGLU with softmax prologue for Qwen3.5 oproj fusion.

Port of Kimi's oproj_topk_v2_swiglu.py adapted for:
- 8-bit quantized weights with gs=64 (Kimi uses 4-bit gs=32)
- Softmax routing (Kimi uses sigmoid)
- E up to 512 with multiple scores per thread (SPT = ceil(E/64))
- Shared expert gate GEMV hidden in TG(0,0,0) SG 0
- Both routed and shared paths use 8-bit quantized weights

Prologue (all TGs):
  Phase 1: distributed x² sum → inv_rms
  Phase 2 (routed TGs only): gate scores → softmax → parallel top-k → norm_topk_prob
  Phase 3 (TG(0,0,0) SG 0): shared_expert_gate 8-bit GEMV → gate_raw

Main SwiGLU (after prologue):
  Routed (z < n_active): 8-bit gate+up+SwiGLU with h_scaled input (inv_rms factored out)
  Shared (z == n_active): 8-bit SwiGLU for shared expert
"""
import mlx.core as mx


def ceil_div(a, b):
    return (a + b - 1) // b


def _gen_oproj_softmax_topk_swiglu_8bit_source(group_size=64, scale_bf16=True,
                                                 n_experts=64, top_k=10,
                                                 norm_topk=True):
    """Generate Metal source for softmax + top-k + SwiGLU with oproj prologue."""
    gs = int(group_size)
    sc_stride = 256 // gs       # groups per K-block (256/64 = 4)
    slid_divisor = gs // 8      # threads per group (64/8 = 8)
    sc_t = "bfloat16_t" if scale_bf16 else "float"
    E = int(n_experts)
    K_TOP = int(top_k)
    SPT = (E + 63) // 64       # scores per thread

    # Score normalization block (TG(0,0,0) thread 0 only)
    if norm_topk:
        score_norm_block = f"""
    // ── norm_topk_prob + write indices (TG(0,0,0) thread 0) ──
    if (tgid.y == 0 && tgid.z == 0 && tid == 0) {{
        float total = 0.0f;
        for (int a = 0; a < {K_TOP}; a++) total += tg_selected_scores[a];
        float inv_total = 1.0f / total;
        for (int a = 0; a < {K_TOP}; a++) {{
            norm_scores[a] = tg_selected_scores[a] * inv_total;
            out_inds[a] = (uint)tg_inds[a];
        }}
    }}"""
    else:
        score_norm_block = f"""
    // ── Write indices and raw scores (TG(0,0,0) thread 0) ──
    if (tgid.y == 0 && tgid.z == 0 && tid == 0) {{
        for (int a = 0; a < {K_TOP}; a++) {{
            norm_scores[a] = tg_selected_scores[a];
            out_inds[a] = (uint)tg_inds[a];
        }}
    }}"""

    return f"""
    const int RESULTS_PER_SG = 4;
    const int E_CONST = {E};
    const int K_TOP_CONST = {K_TOP};
    const int SPT = {SPT};

    int N_INTER = N_INTER_val;
    int SHARED_INTER = SHARED_INTER_val;
    int K = K_val;
    int n_active = n_active_val;
    int N_OPROJ_TG = N_OPROJ_TG_val;

    uint3 tgid = threadgroup_position_in_grid;
    uint sgid = simdgroup_index_in_threadgroup;  // 0 or 1
    uint slid = thread_index_in_simdgroup;       // 0..31
    int tid = int(sgid) * 32 + int(slid);        // 0..63

    // ═══════════════════════════════════════════════════════════════
    // PROLOGUE PHASE 1: distributed x² sum → inv_rms (ALL TGs)
    // ═══════════════════════════════════════════════════════════════
    int chunk = (N_OPROJ_TG + 63) / 64;
    int x2_start = tid * chunk;
    int x2_end = min(x2_start + chunk, N_OPROJ_TG);
    float local_x2 = 0.0f;
    for (int i = x2_start; i < x2_end; i++) local_x2 += x2_partials[i];
    float sg_x2_sum = simd_sum(local_x2);

    threadgroup float tg_x2_sg[2];
    if (slid == 0) tg_x2_sg[sgid] = sg_x2_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total_x2 = tg_x2_sg[0] + tg_x2_sg[1];
    float inv_rms = metal::precise::rsqrt(total_x2 / (float)K + 1e-6f);

    // ═══════════════════════════════════════════════════════════════
    // PROLOGUE PHASE 2: Softmax + Top-k (routed TGs only)
    // ═══════════════════════════════════════════════════════════════
    threadgroup int tg_inds[{K_TOP}];
    threadgroup float tg_selected_scores[{K_TOP}];

    if (tgid.z < (uint)n_active) {{
        // Load gate scores and apply inv_rms
        float my_scores[SPT];
        for (int j = 0; j < SPT; j++) {{
            int e = tid * SPT + j;
            if (e < E_CONST)
                my_scores[j] = (gate_part_a[e] + gate_part_b[e]) * inv_rms;
            else
                my_scores[j] = -1e30f;
        }}

        // ── Softmax: distributed max ──
        float local_max = -1e30f;
        for (int j = 0; j < SPT; j++)
            local_max = max(local_max, my_scores[j]);
        float sg_max_val = simd_max(local_max);
        threadgroup float tg_softmax_sg[2];
        if (slid == 0) tg_softmax_sg[sgid] = sg_max_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tg_max = max(tg_softmax_sg[0], tg_softmax_sg[1]);

        // ── Softmax: exp + distributed sum ──
        float local_sum = 0.0f;
        for (int j = 0; j < SPT; j++) {{
            float e_val = metal::exp(my_scores[j] - tg_max);
            my_scores[j] = e_val;
            local_sum += e_val;
        }}
        float sg_sum_val = simd_sum(local_sum);
        if (slid == 0) tg_softmax_sg[sgid] = sg_sum_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tg_sum = tg_softmax_sg[0] + tg_softmax_sg[1];

        // ── Softmax: normalize ──
        float inv_sum = 1.0f / tg_sum;
        for (int j = 0; j < SPT; j++)
            my_scores[j] *= inv_sum;

        // ── Parallel top-k: K_TOP rounds ──
        threadgroup float tg_tk_val[2];
        threadgroup int tg_tk_info[2];

        for (int round = 0; round < K_TOP_CONST; round++) {{
            // Find local best among SPT scores
            float best = -1.0f;
            int best_e = -1;
            for (int j = 0; j < SPT; j++) {{
                int e = tid * SPT + j;
                if (e < E_CONST && my_scores[j] > best) {{
                    best = my_scores[j];
                    best_e = e;
                }}
            }}

            // SG-level max + winner identification
            float sg_best = simd_max(best);
            int candidate = (best == sg_best && best > 0.0f) ? int(slid) : 999;
            int sg_winner = simd_min(candidate);

            if (slid == 0) {{
                tg_tk_val[sgid] = sg_best;
                tg_tk_info[sgid] = sg_winner;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Determine global winner (SG with higher max; tie-break: SG 0)
            int winner_sg = (tg_tk_val[0] >= tg_tk_val[1]) ? 0 : 1;
            int winner_lane = tg_tk_info[winner_sg];
            int winner_tid = winner_sg * 32 + winner_lane;

            // Winner writes expert index and score to TG memory
            if (tid == winner_tid) {{
                tg_inds[round] = best_e;
                tg_selected_scores[round] = best;
                // Disable the winning score in register
                for (int j = 0; j < SPT; j++) {{
                    if (tid * SPT + j == best_e) {{
                        my_scores[j] = -1.0f;
                        break;
                    }}
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
    }}
{score_norm_block}

    // ═══════════════════════════════════════════════════════════════
    // PROLOGUE PHASE 3: Shared expert gate (TG(0,0,0) SG 0 only)
    // 8-bit GEMV: W_seg (1, K) × X (K,) → gate_raw scalar
    // Input is h_scaled; multiply result by inv_rms for true gate value
    // ═══════════════════════════════════════════════════════════════
    if (tgid.y == 0 && tgid.z == 0 && sgid == 0) {{
        const int VPT = 8;
        const int BLOCK = 256;  // 32 * VPT
        int K_groups_seg = K / {gs};

        const device uint8_t* wg_seg = (const device uint8_t*)W_seg
            + slid * VPT;
        const device {sc_t}* sc_seg = (const device {sc_t}*)S_seg
            + slid / {slid_divisor};
        const device {sc_t}* bi_seg = (const device {sc_t}*)B_seg
            + slid / {slid_divisor};
        int xb = slid * VPT;

        float gate_acc = 0.0f;
        for (int k = 0; k < K; k += BLOCK) {{
            float xsum = 0.0f, wacc = 0.0f;
            for (int i = 0; i < VPT; i++) {{
                float xi = float(X[xb + i]);
                xsum += xi;
                wacc += xi * float(wg_seg[i]);
            }}
            gate_acc += float(*sc_seg) * wacc + xsum * float(*bi_seg);
            wg_seg += BLOCK;
            sc_seg += {sc_stride};
            bi_seg += {sc_stride};
            xb += BLOCK;
        }}
        gate_acc = simd_sum(gate_acc);
        if (slid == 0) gate_raw[0] = gate_acc * inv_rms;
    }}

    // ═══════════════════════════════════════════════════════════════
    // MAIN SwiGLU BODY
    // ═══════════════════════════════════════════════════════════════
    int out_row = tgid.y * 8 + sgid * RESULTS_PER_SG;
    int row_limit = (tgid.z < (uint)n_active) ? N_INTER : SHARED_INTER;
    if (out_row >= row_limit) return;

    float gate_result[4] = {{0, 0, 0, 0}};
    float up_result[4] = {{0, 0, 0, 0}};

    if (tgid.z < (uint)n_active) {{
        // ═══════ 8-BIT ROUTED EXPERT PATH ═══════
        const int VALUES_PER_THREAD = 8;
        const int BLOCK_SIZE = 256;

        int N_TOTAL = 2 * N_INTER;
        int K_groups = K / {gs};

        int expert = tg_inds[tgid.z];

        const device uint8_t* ws_gate = (const device uint8_t*)W
            + (long)expert * N_TOTAL * K + out_row * K + slid * VALUES_PER_THREAD;
        const device {sc_t}* sc_gate = (const device {sc_t}*)S
            + (long)expert * N_TOTAL * K_groups + out_row * K_groups
            + slid / {slid_divisor};
        const device {sc_t}* bi_gate = (const device {sc_t}*)B_q
            + (long)expert * N_TOTAL * K_groups + out_row * K_groups
            + slid / {slid_divisor};

        const device uint8_t* ws_up = (const device uint8_t*)W
            + (long)expert * N_TOTAL * K + (out_row + N_INTER) * K
            + slid * VALUES_PER_THREAD;
        const device {sc_t}* sc_up = (const device {sc_t}*)S
            + (long)expert * N_TOTAL * K_groups + (out_row + N_INTER) * K_groups
            + slid / {slid_divisor};
        const device {sc_t}* bi_up = (const device {sc_t}*)B_q
            + (long)expert * N_TOTAL * K_groups + (out_row + N_INTER) * K_groups
            + slid / {slid_divisor};

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
                for (int i = 0; i < 8; i++)
                    accum_g += x_thread[i] * float(wg[i]);
                gate_result[row] += sg * accum_g + xsum * bg;

                // Up projection
                const device uint8_t* wu = ws_up + row * K;
                float su = float(sc_up[row * K_groups]);
                float bu = float(bi_up[row * K_groups]);
                float accum_u = 0;
                for (int i = 0; i < 8; i++)
                    accum_u += x_thread[i] * float(wu[i]);
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

        // Epilogue: apply inv_rms (factored), SwiGLU, write f32
        device float* yp = Y_routed + tgid.z * N_INTER + out_row;
        for (int row = 0; row < RESULTS_PER_SG; row++) {{
            float g = simd_sum(gate_result[row]) * inv_rms;
            float u = simd_sum(up_result[row]) * inv_rms;
            if (slid == 0) {{
                float silu_g = g / (1.0f + metal::exp(-g));
                yp[row] = silu_g * u;
            }}
        }}

    }} else {{
        // ═══════ 8-BIT SHARED EXPERT PATH ═══════
        const int VALUES_PER_THREAD = 8;
        const int BLOCK_SIZE = 256;

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
                const device uint8_t* wg_s = ws_gate + row * K;
                float sg_s = float(sc_gate[row * K_groups]);
                float bg_s = float(bi_gate[row * K_groups]);
                float accum_g = 0;
                for (int i = 0; i < 8; i++)
                    accum_g += x_thread[i] * float(wg_s[i]);
                gate_result[row] += sg_s * accum_g + xsum * bg_s;

                const device uint8_t* wu_s = ws_up + row * K;
                float su_s = float(sc_up[row * K_groups]);
                float bu_s = float(bi_up[row * K_groups]);
                float accum_u = 0;
                for (int i = 0; i < 8; i++)
                    accum_u += x_thread[i] * float(wu_s[i]);
                up_result[row] += su_s * accum_u + xsum * bu_s;
            }}

            ws_gate += BLOCK_SIZE;
            ws_up += BLOCK_SIZE;
            sc_gate += {sc_stride};
            sc_up += {sc_stride};
            bi_gate += {sc_stride};
            bi_up += {sc_stride};
            x_base += BLOCK_SIZE;
        }}

        // Epilogue: apply inv_rms (factored), SwiGLU, write f32
        device float* yp = Y_shared + out_row;
        for (int tm = 0; tm < RESULTS_PER_SG; tm++) {{
            float g = simd_sum(gate_result[tm]) * inv_rms;
            float u = simd_sum(up_result[tm]) * inv_rms;
            if (slid == 0) {{
                float silu_g = g / (1.0f + metal::exp(-g));
                yp[tm] = silu_g * u;
            }}
        }}
    }}
"""


_oproj_softmax_swiglu_8bit_kernels = {}


def _get_oproj_softmax_swiglu_8bit_kernel(group_size=64, scale_bf16=True,
                                            n_experts=64, top_k=10,
                                            norm_topk=True):
    key = (group_size, scale_bf16, n_experts, top_k, norm_topk)
    if key not in _oproj_softmax_swiglu_8bit_kernels:
        sc_tag = "_bf16sc" if scale_bf16 else ""
        nt_tag = "_nt" if norm_topk else ""
        _oproj_softmax_swiglu_8bit_kernels[key] = mx.fast.metal_kernel(
            name=(f"oproj_softmax_topk_swiglu_8bit_gs{group_size}"
                  f"_e{n_experts}_k{top_k}{sc_tag}{nt_tag}"),
            input_names=[
                "W", "S", "B_q",                    # routed expert weights
                "W_shared", "S_shared", "B_shared",  # shared expert weights
                "X",                                  # h_scaled (K,) bf16
                "gate_part_a", "gate_part_b",         # (E,) f32
                "x2_partials",                        # (N_OPROJ_TG,) f32
                "W_seg", "S_seg", "B_seg",            # shared_expert_gate weights
                "N_INTER_val", "SHARED_INTER_val",
                "K_val", "n_active_val", "N_OPROJ_TG_val",
            ],
            output_names=["Y_routed", "Y_shared", "out_inds",
                          "norm_scores", "gate_raw"],
            source=_gen_oproj_softmax_topk_swiglu_8bit_source(
                group_size, scale_bf16, n_experts, top_k, norm_topk),
        )
    return _oproj_softmax_swiglu_8bit_kernels[key]


def fused_oproj_softmax_topk_swiglu_8bit(
    w_q, s, b_q,                         # routed expert weights
    w_shared, s_shared, b_shared,         # shared expert weights
    h_scaled,                             # (K,) bf16 — h * w_rms
    gate_part_a, gate_part_b,             # (E,) f32 — gate decomposition
    x2_partials,                          # (N_OPROJ_TG,) f32 — per-TG x²
    w_seg, s_seg, b_seg,                  # shared_expert_gate 8-bit weights
    n_inter, k_hidden,                    # MoE dimensions
    n_experts, top_k,                     # routing params
    n_oproj_tg,                           # number of o_proj TGs (for x²)
    group_size=64,
    shared_inter=None,
    norm_topk=True,
):
    """Single-dispatch softmax + top-k + merged 8-bit SwiGLU with oproj prologue.

    Prologue:
      Phase 1: distributed x² → inv_rms (all TGs)
      Phase 2: softmax(gate_part_a + gate_part_b) → top-k → norm (routed TGs)
      Phase 3: shared_expert_gate 8-bit GEMV → gate_raw (TG(0,0,0) SG 0)

    Main: 8-bit gate+up+SwiGLU for routed + shared experts.
    inv_rms is factored out of the inner loop and applied once in epilogue.

    Args:
        h_scaled: (K,) bf16 — h * w_rms from Dispatch 1
        gate_part_a: (E,) f32 — M1 @ attn_out from Dispatch 1
        gate_part_b: (E,) f32 — W_fused @ residual from Dispatch 1
        x2_partials: (N_OPROJ_TG,) f32 — per-TG Σh² from Dispatch 1
        w_seg/s_seg/b_seg: shared_expert_gate 8-bit weights (1, K/4) uint32

    Returns:
        (Y_routed, Y_shared, out_inds, norm_scores, gate_raw):
            Y_routed: (top_k, n_inter) f32
            Y_shared: (shared_inter,) f32
            out_inds: (top_k,) uint32
            norm_scores: (top_k,) f32
            gate_raw: (1,) f32 — raw shared expert gate value (sigmoid in epilogue)
    """
    scale_bf16 = (s.dtype == mx.bfloat16)
    kern = _get_oproj_softmax_swiglu_8bit_kernel(
        group_size, scale_bf16, n_experts, top_k, norm_topk)

    n_inter_val = int(n_inter)
    shared_inter_val = int(shared_inter) if shared_inter is not None else n_inter_val
    n_active = int(top_k)
    max_inter = max(n_inter_val, shared_inter_val)

    results = kern(
        inputs=[
            w_q, s, b_q,
            w_shared, s_shared, b_shared,
            h_scaled,
            gate_part_a, gate_part_b,
            x2_partials,
            w_seg, s_seg, b_seg,
            n_inter_val, shared_inter_val,
            int(k_hidden), n_active, int(n_oproj_tg),
        ],
        output_shapes=[
            (n_active, n_inter_val),     # Y_routed
            (shared_inter_val,),         # Y_shared
            (n_active,),                 # out_inds
            (n_active,),                 # norm_scores
            (1,),                        # gate_raw
        ],
        output_dtypes=[
            mx.float32,    # Y_routed
            mx.float32,    # Y_shared
            mx.uint32,     # out_inds
            mx.float32,    # norm_scores
            mx.float32,    # gate_raw
        ],
        grid=(32, ceil_div(max_inter, 8) * 2, n_active + 1),
        threadgroup=(32, 2, 1),
    )
    return results[0], results[1], results[2], results[3], results[4]
