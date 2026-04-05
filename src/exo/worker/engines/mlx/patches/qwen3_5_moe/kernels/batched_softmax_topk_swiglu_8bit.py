"""Dispatch 2 (batched): Softmax prologue + 8-bit SwiGLU for Qwen3.5 MoE.

Combines the prologue from oproj_softmax_topk_swiglu_8bit (B=1) with the
batched body from batched_merged_swiglu_8bit. All constants are baked into
the Metal source at Python code-generation time.

Grid z-dimension: B * n_active + 1
  - tgid.z < B * n_active: routed expert TGs
    batch_id = tgid.z / n_active, local_z = tgid.z % n_active
  - tgid.z == B * n_active: shared expert TG (register-level weight sharing
    for B batch elements, including shared_expert_gate GEMV)

Prologue (all TGs):
  Phase 1: distributed x2 partial sum -> inv_rms (per batch_id)
  Phase 2 (routed TGs only): gate scores -> softmax -> parallel top-k ->
           norm_topk_prob -> write out_inds / norm_scores
  Phase 3 (shared TG, SG 0): shared_expert_gate 8-bit GEMV with
           register-level weight sharing -> gate_raw[B]

Body (after TG barrier):
  Routed: 8-bit gate+up+SwiGLU with h_scaled[batch_id] input
  Shared: register-level weight sharing for B batch elements
"""
import mlx.core as mx

from ..common import METAL_HALF_TYPE


def ceil_div(a, b):
    return (a + b - 1) // b


def _gen_batched_softmax_topk_swiglu_source(
    N_INTER, SHARED_INTER, K, n_active, B,
    n_experts=256, top_k=10, norm_topk=True, group_size=64,
    n_oproj_tg=64,
):
    """Generate Metal source for batched softmax + top-k + SwiGLU.

    All routed TGs compute the full softmax+topk for their batch_id into TG
    memory. Only the TG with local_z==0 writes out_inds/norm_scores to device
    memory. Each routed TG reads its own expert from tg_inds[local_z].
    """
    gs = group_size
    sc_stride = 256 // gs
    slid_divisor = gs // 8
    N_TOTAL = 2 * N_INTER
    K_groups = K // gs
    SHARED_K_groups = K // gs
    total_routed = B * n_active
    E = int(n_experts)
    K_TOP = int(top_k)
    SPT = (E + 63) // 64

    # ── Shared expert body: unrolled per-batch code ──
    shared_x_load = "\n".join(f"""
            float x{b}_thread[VALUES_PER_THREAD];
            float xsum{b} = 0;
            for (int i = 0; i < VALUES_PER_THREAD; i++) {{
                float xi = float(X[{b} * K_DIM + x_base + i]);
                x{b}_thread[i] = xi;
                xsum{b} += xi;
            }}""" for b in range(B))

    shared_gate_qdot = "\n".join(f"""
                float accum_g{b} = 0;
                for (int i = 0; i < VALUES_PER_THREAD; i++) accum_g{b} += x{b}_thread[i] * wg_vals[i];
                gate{b}[row] += sg * accum_g{b} + xsum{b} * bg;""" for b in range(B))

    shared_up_qdot = "\n".join(f"""
                float accum_u{b} = 0;
                for (int i = 0; i < VALUES_PER_THREAD; i++) accum_u{b} += x{b}_thread[i] * wu_vals[i];
                up{b}[row] += su * accum_u{b} + xsum{b} * bu;""" for b in range(B))

    shared_result_decls = "\n        ".join(
        f"float gate{b}[RESULTS_PER_SG] = {{0,0,0,0}}; float up{b}[RESULTS_PER_SG] = {{0,0,0,0}};"
        for b in range(B))

    shared_write_lines = []
    for b in range(B):
        shared_write_lines.append(f"""
        for (int row = 0; row < RESULTS_PER_SG; row++) {{
            float g{b} = simd_sum(gate{b}[row]) * inv_rms_{b};
            float u{b} = simd_sum(up{b}[row]) * inv_rms_{b};
            if (slid == 0) {{
                float silu_g{b} = g{b} / (1.0f + metal::exp(-g{b}));
                Y_shared[{b} * SHARED_INTER_DIM + out_row + row] = silu_g{b} * u{b};
            }}
        }}""")
    shared_write = "\n".join(shared_write_lines)

    # inv_rms for all B batches in the shared TG
    shared_inv_rms_lines = []
    for b in range(B):
        shared_inv_rms_lines.append(f"""
        float local_x2_{b} = 0.0f;
        for (int i = x2_start; i < x2_end; i++) local_x2_{b} += x2_partials[{b} * N_OPROJ_TG_DIM + i];
        float sg_x2_{b} = simd_sum(local_x2_{b});
        if (slid == 0) tg_x2_sg[sgid] = sg_x2_{b};
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv_rms_{b} = metal::precise::rsqrt((tg_x2_sg[0] + tg_x2_sg[1]) / (float)K_DIM + 1e-6f);""")
    shared_inv_rms_block = "\n".join(shared_inv_rms_lines)

    # Shared expert gate GEMV accumulator declarations
    seg_acc_decls = "\n        ".join(
        f"float seg_gate_acc{b} = 0.0f;" for b in range(B))

    seg_write = "\n".join(
        f"            gate_raw[{b}] = seg_gate_acc{b} * inv_rms_{b};"
        for b in range(B))

    return f"""
    const int RESULTS_PER_SG = 4;
    const int VALUES_PER_THREAD = 8;
    const int BLOCK_SIZE = 256;
    const int N_INTER_DIM = {N_INTER};
    const int SHARED_INTER_DIM = {SHARED_INTER};
    const int K_DIM = {K};
    const int K_GROUPS = {K_groups};
    const int N_TOTAL = {N_TOTAL};
    const int N_ACTIVE = {n_active};
    const int TOTAL_ROUTED = {total_routed};
    const int BATCH_SIZE = {B};
    const int E_CONST = {E};
    const int K_TOP_CONST = {K_TOP};
    const int SPT = {SPT};
    const int N_OPROJ_TG_DIM = {n_oproj_tg};

    uint3 tgid = threadgroup_position_in_grid;
    uint sgid = simdgroup_index_in_threadgroup;  // 0 or 1
    uint slid = thread_index_in_simdgroup;       // 0..31
    int tid = int(sgid) * 32 + int(slid);        // 0..63

    threadgroup float tg_x2_sg[2];
    threadgroup int tg_inds[{K_TOP}];
    threadgroup float tg_selected_scores[{K_TOP}];

    if (tgid.z < (uint)TOTAL_ROUTED) {{
        // ═══════════════════════════════════════════════════════════════
        // ROUTED TG PROLOGUE
        // ═══════════════════════════════════════════════════════════════
        int flat_idx = (int)tgid.z;
        int batch_id = flat_idx / N_ACTIVE;
        int local_z = flat_idx % N_ACTIVE;

        // Phase 1: distributed x2 sum -> inv_rms for batch_id
        int chunk = (N_OPROJ_TG_DIM + 63) / 64;
        int x2_start = tid * chunk;
        int x2_end = min(x2_start + chunk, N_OPROJ_TG_DIM);
        float local_x2 = 0.0f;
        for (int i = x2_start; i < x2_end; i++)
            local_x2 += x2_partials[batch_id * N_OPROJ_TG_DIM + i];
        float sg_x2_sum = simd_sum(local_x2);
        if (slid == 0) tg_x2_sg[sgid] = sg_x2_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total_x2 = tg_x2_sg[0] + tg_x2_sg[1];
        float inv_rms = metal::precise::rsqrt(total_x2 / (float)K_DIM + 1e-6f);

        // Phase 2: ALL routed TGs compute full softmax + top-k for their
        // batch_id. Each TG gets its own copy in TG memory (tg_inds,
        // tg_selected_scores). This avoids cross-TG communication.
        {{
            float my_scores[SPT];
            for (int j = 0; j < SPT; j++) {{
                int e = tid * SPT + j;
                if (e < E_CONST)
                    my_scores[j] = (gate_part_a[batch_id * E_CONST + e]
                                  + gate_part_b[batch_id * E_CONST + e]) * inv_rms;
                else
                    my_scores[j] = -1e30f;
            }}

            // Softmax: distributed max
            float local_max = -1e30f;
            for (int j = 0; j < SPT; j++)
                local_max = max(local_max, my_scores[j]);
            float sg_max_val = simd_max(local_max);
            threadgroup float tg_softmax_sg[2];
            if (slid == 0) tg_softmax_sg[sgid] = sg_max_val;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float tg_max = max(tg_softmax_sg[0], tg_softmax_sg[1]);

            // Softmax: exp + distributed sum
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

            // Softmax: normalize
            float inv_sum = 1.0f / tg_sum;
            for (int j = 0; j < SPT; j++)
                my_scores[j] *= inv_sum;

            // Parallel top-k: K_TOP rounds
            threadgroup float tg_tk_val[2];
            threadgroup int tg_tk_info[2];

            for (int round = 0; round < K_TOP_CONST; round++) {{
                float best = -1.0f;
                int best_e = -1;
                for (int j = 0; j < SPT; j++) {{
                    int e = tid * SPT + j;
                    if (e < E_CONST && my_scores[j] > best) {{
                        best = my_scores[j];
                        best_e = e;
                    }}
                }}

                float sg_best = simd_max(best);
                int candidate = (best == sg_best && best > 0.0f) ? int(slid) : 999;
                int sg_winner = simd_min(candidate);

                if (slid == 0) {{
                    tg_tk_val[sgid] = sg_best;
                    tg_tk_info[sgid] = sg_winner;
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);

                int winner_sg = (tg_tk_val[0] >= tg_tk_val[1]) ? 0 : 1;
                int winner_lane = tg_tk_info[winner_sg];
                int winner_tid = winner_sg * 32 + winner_lane;

                if (tid == winner_tid) {{
                    tg_inds[round] = best_e;
                    tg_selected_scores[round] = best;
                    for (int j = 0; j < SPT; j++) {{
                        if (tid * SPT + j == best_e) {{
                            my_scores[j] = -1.0f;
                            break;
                        }}
                    }}
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}

            // Only local_z==0 writes norm_scores + out_inds to device memory
            if (local_z == 0 && tid == 0) {{
                float total_score = 0.0f;
                for (int a = 0; a < {K_TOP}; a++) total_score += tg_selected_scores[a];
                float inv_total = {"1.0f / total_score" if norm_topk else "1.0f"};
                for (int a = 0; a < {K_TOP}; a++) {{
                    norm_scores[batch_id * {K_TOP} + a] = tg_selected_scores[a] * inv_total;
                    out_inds[batch_id * {K_TOP} + a] = (uint)tg_inds[a];
                }}
            }}
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ═══════════════════════════════════════════════════════════════
        // ROUTED BODY: 8-bit gate+up+SwiGLU
        // ═══════════════════════════════════════════════════════════════
        int out_row = tgid.y * 8 + sgid * RESULTS_PER_SG;
        if (out_row >= N_INTER_DIM) return;

        int expert = tg_inds[local_z];

        const device uint8_t* ws_gate = (const device uint8_t*)W
            + (long)expert * N_TOTAL * K_DIM + out_row * K_DIM + slid * VALUES_PER_THREAD;
        const device bfloat16_t* sc_gate = (const device bfloat16_t*)S
            + (long)expert * N_TOTAL * K_GROUPS + out_row * K_GROUPS + slid / {slid_divisor};
        const device bfloat16_t* bi_gate = (const device bfloat16_t*)B_q
            + (long)expert * N_TOTAL * K_GROUPS + out_row * K_GROUPS + slid / {slid_divisor};

        const device uint8_t* ws_up = (const device uint8_t*)W
            + (long)expert * N_TOTAL * K_DIM + (out_row + N_INTER_DIM) * K_DIM + slid * VALUES_PER_THREAD;
        const device bfloat16_t* sc_up = (const device bfloat16_t*)S
            + (long)expert * N_TOTAL * K_GROUPS + (out_row + N_INTER_DIM) * K_GROUPS + slid / {slid_divisor};
        const device bfloat16_t* bi_up = (const device bfloat16_t*)B_q
            + (long)expert * N_TOTAL * K_GROUPS + (out_row + N_INTER_DIM) * K_GROUPS + slid / {slid_divisor};

        int x_base = batch_id * K_DIM + slid * VALUES_PER_THREAD;

        float gate_result[4] = {{0, 0, 0, 0}};
        float up_result[4] = {{0, 0, 0, 0}};

        for (int k = 0; k < K_DIM; k += BLOCK_SIZE) {{
            float x_thread[8];
            float xsum = 0;
            for (int i = 0; i < 8; i++) {{
                float xi = float(X[x_base + i]);
                x_thread[i] = xi;
                xsum += xi;
            }}

            for (int row = 0; row < RESULTS_PER_SG; row++) {{
                const device uint8_t* wg = ws_gate + row * K_DIM;
                float sg = float(sc_gate[row * K_GROUPS]);
                float bg = float(bi_gate[row * K_GROUPS]);
                float accum_g = 0;
                for (int i = 0; i < 8; i++) accum_g += x_thread[i] * float(wg[i]);
                gate_result[row] += sg * accum_g + xsum * bg;

                const device uint8_t* wu = ws_up + row * K_DIM;
                float su = float(sc_up[row * K_GROUPS]);
                float bu = float(bi_up[row * K_GROUPS]);
                float accum_u = 0;
                for (int i = 0; i < 8; i++) accum_u += x_thread[i] * float(wu[i]);
                up_result[row] += su * accum_u + xsum * bu;
            }}

            ws_gate += BLOCK_SIZE; ws_up += BLOCK_SIZE;
            sc_gate += {sc_stride}; sc_up += {sc_stride};
            bi_gate += {sc_stride}; bi_up += {sc_stride};
            x_base += BLOCK_SIZE;
        }}

        // Epilogue: apply inv_rms (factored), SwiGLU, write f32
        device float* yp = Y_routed + flat_idx * N_INTER_DIM + out_row;
        for (int row = 0; row < RESULTS_PER_SG; row++) {{
            float g = simd_sum(gate_result[row]) * inv_rms;
            float u = simd_sum(up_result[row]) * inv_rms;
            if (slid == 0) {{
                float silu_g = g / (1.0f + metal::exp(-g));
                yp[row] = silu_g * u;
            }}
        }}

    }} else {{
        // ═══════════════════════════════════════════════════════════════
        // SHARED TG (tgid.z == TOTAL_ROUTED)
        // ═══════════════════════════════════════════════════════════════

        // Phase 1: compute inv_rms for ALL B batch elements
        int chunk = (N_OPROJ_TG_DIM + 63) / 64;
        int x2_start = tid * chunk;
        int x2_end = min(x2_start + chunk, N_OPROJ_TG_DIM);
{shared_inv_rms_block}

        // Phase 3: shared_expert_gate 8-bit GEMV (SG 0 only)
        // Load W_seg once, compute B dot products via register-level weight sharing
        if (sgid == 0) {{
            const int VPT = 8;
            const int SEG_BLOCK = 256;  // 32 * VPT

            const device uint8_t* seg_w_ptr = (const device uint8_t*)W_seg
                + slid * VPT;
            const device bfloat16_t* seg_sc = (const device bfloat16_t*)S_seg
                + slid / {slid_divisor};
            const device bfloat16_t* seg_bi = (const device bfloat16_t*)B_seg
                + slid / {slid_divisor};
            int seg_xb = slid * VPT;

            {seg_acc_decls}

            for (int k = 0; k < K_DIM; k += SEG_BLOCK) {{
                // Load weight block once into registers
                float seg_w_regs[VPT];
                for (int i = 0; i < VPT; i++) seg_w_regs[i] = float(seg_w_ptr[i]);
                float seg_sc_val = float(*seg_sc);
                float seg_bi_val = float(*seg_bi);

                // Compute B dot products from the same weight registers
{chr(10).join(f'''                {{
                    float xsum{b} = 0.0f, wacc{b} = 0.0f;
                    for (int i = 0; i < VPT; i++) {{
                        float xi = float(X[{b} * K_DIM + seg_xb + i]);
                        xsum{b} += xi;
                        wacc{b} += xi * seg_w_regs[i];
                    }}
                    seg_gate_acc{b} += seg_sc_val * wacc{b} + xsum{b} * seg_bi_val;
                }}''' for b in range(B))}

                seg_w_ptr += SEG_BLOCK;
                seg_sc += {sc_stride};
                seg_bi += {sc_stride};
                seg_xb += SEG_BLOCK;
            }}

            // Reduce across SG and write gate_raw[B]
{chr(10).join(f"            seg_gate_acc{b} = simd_sum(seg_gate_acc{b});" for b in range(B))}
            if (slid == 0) {{
{seg_write}
            }}
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ═══════════════════════════════════════════════════════════════
        // SHARED BODY: register-level weight sharing for B batch elements
        // ═══════════════════════════════════════════════════════════════
        int out_row = tgid.y * 8 + sgid * RESULTS_PER_SG;
        if (out_row >= SHARED_INTER_DIM) return;

        const device uint8_t* ws_gate = (const device uint8_t*)W_shared
            + (long)out_row * K_DIM + slid * VALUES_PER_THREAD;
        const device bfloat16_t* sc_gate = (const device bfloat16_t*)S_shared
            + (long)out_row * {SHARED_K_groups} + slid / {slid_divisor};
        const device bfloat16_t* bi_gate = (const device bfloat16_t*)B_shared
            + (long)out_row * {SHARED_K_groups} + slid / {slid_divisor};

        const device uint8_t* ws_up = (const device uint8_t*)W_shared
            + (long)(out_row + SHARED_INTER_DIM) * K_DIM + slid * VALUES_PER_THREAD;
        const device bfloat16_t* sc_up = (const device bfloat16_t*)S_shared
            + (long)(out_row + SHARED_INTER_DIM) * {SHARED_K_groups} + slid / {slid_divisor};
        const device bfloat16_t* bi_up = (const device bfloat16_t*)B_shared
            + (long)(out_row + SHARED_INTER_DIM) * {SHARED_K_groups} + slid / {slid_divisor};

        int x_base = slid * VALUES_PER_THREAD;
        {shared_result_decls}

        for (int k = 0; k < K_DIM; k += BLOCK_SIZE) {{
            // Load x for all {B} batch elements
{shared_x_load}

            for (int row = 0; row < RESULTS_PER_SG; row++) {{
                // Load gate weights once into registers
                const device uint8_t* wg = ws_gate + row * K_DIM;
                float sg = float(sc_gate[row * {SHARED_K_groups}]);
                float bg = float(bi_gate[row * {SHARED_K_groups}]);
                float wg_vals[VALUES_PER_THREAD];
                for (int i = 0; i < VALUES_PER_THREAD; i++) wg_vals[i] = float(wg[i]);

                // Compute gate for all {B} batches from registers
{shared_gate_qdot}

                // Load up weights once into registers
                const device uint8_t* wu = ws_up + row * K_DIM;
                float su = float(sc_up[row * {SHARED_K_groups}]);
                float bu = float(bi_up[row * {SHARED_K_groups}]);
                float wu_vals[VALUES_PER_THREAD];
                for (int i = 0; i < VALUES_PER_THREAD; i++) wu_vals[i] = float(wu[i]);

                // Compute up for all {B} batches from registers
{shared_up_qdot}
            }}

            ws_gate += BLOCK_SIZE; ws_up += BLOCK_SIZE;
            sc_gate += {sc_stride}; sc_up += {sc_stride};
            bi_gate += {sc_stride}; bi_up += {sc_stride};
            x_base += BLOCK_SIZE;
        }}

        // SwiGLU epilogue + write for all {B} batches (with inv_rms)
{shared_write}
    }}
"""

_batched_softmax_topk_swiglu_cache = {}


def _get_batched_softmax_topk_swiglu_kernel(
    N_INTER, SHARED_INTER, K, n_active, B,
    n_experts=256, top_k=10, norm_topk=True, group_size=64,
    n_oproj_tg=64,
):
    key = (N_INTER, SHARED_INTER, K, n_active, B, n_experts, top_k, norm_topk, group_size, n_oproj_tg)
    if key not in _batched_softmax_topk_swiglu_cache:
        nt_tag = "_nt" if norm_topk else ""
        _batched_softmax_topk_swiglu_cache[key] = mx.fast.metal_kernel(
            name=(f"batched_softmax_topk_swiglu_8bit"
                  f"_NI{N_INTER}_SI{SHARED_INTER}_K{K}"
                  f"_na{n_active}_B{B}_E{n_experts}_k{top_k}"
                  f"_gs{group_size}{nt_tag}"),
            input_names=[
                "W", "S", "B_q",                    # routed expert weights
                "W_shared", "S_shared", "B_shared",  # shared expert weights
                "W_seg", "S_seg", "B_seg",            # shared_expert_gate weights
                "X",                                  # h_scaled (B, K) bf16
                "gate_part_a", "gate_part_b",         # (B, E) f32
                "x2_partials",                        # (B, N_OPROJ_TG) f32
            ],
            output_names=["Y_routed", "Y_shared", "out_inds",
                          "norm_scores", "gate_raw"],
            source=_gen_batched_softmax_topk_swiglu_source(
                N_INTER, SHARED_INTER, K, n_active, B,
                n_experts, top_k, norm_topk, group_size, n_oproj_tg,
            ).replace("bfloat16_t", METAL_HALF_TYPE),
        )
    return _batched_softmax_topk_swiglu_cache[key]


def batched_softmax_topk_swiglu_8bit(
    w_gu, s_gu, b_gu,               # routed gate+up weights (E, 2*N_INTER, K/4)
    w_shared, s_shared, b_shared,    # shared gate+up weights (2*SHARED_INTER, K/4)
    w_seg, s_seg, b_seg,             # shared_expert_gate weights (1, K/4)
    h_scaled,                        # (B, K) bf16 — from Dispatch 1
    gate_part_a,                     # (B, E) f32 — from Dispatch 1
    gate_part_b,                     # (B, E) f32 — from Dispatch 1
    x2_partials,                     # (B, N_OPROJ_TG) f32 — from Dispatch 1
    n_inter, k_hidden, batch_size, n_active,
    n_oproj_tg, n_experts=256,
    shared_inter=None, group_size=64,
):
    """Batched Dispatch 2: softmax + top-k + merged 8-bit SwiGLU with oproj prologue.

    Prologue (per-batch):
      Phase 1: distributed x2 -> inv_rms (all TGs, indexed by batch_id)
      Phase 2: softmax(gate_scores) -> top-k -> norm_topk_prob (all routed TGs)
      Phase 3: shared_expert_gate 8-bit GEMV -> gate_raw[B] (shared TG, SG 0)

    Body:
      Routed: 8-bit gate+up+SwiGLU with h_scaled[batch_id] input, inv_rms factored
      Shared: register-level weight sharing for B batch elements

    Args:
        w_gu: stacked routed weights (E, 2*N_INTER, K/4) uint32
        s_gu: routed scales (E, 2*N_INTER, K/gs) bf16
        b_gu: routed biases (E, 2*N_INTER, K/gs) bf16
        w_shared: shared gate+up stacked (2*SHARED_INTER, K/4) uint32
        s_shared: shared scales (2*SHARED_INTER, K/gs) bf16
        b_shared: shared biases (2*SHARED_INTER, K/gs) bf16
        w_seg/s_seg/b_seg: shared_expert_gate 8-bit weights (1, K/4) uint32
        h_scaled: (B, K) bf16 — h * w_rms from Dispatch 1
        gate_part_a: (B, E) f32 — partial gate scores from Dispatch 1
        gate_part_b: (B, E) f32 — partial gate scores from Dispatch 1
        x2_partials: (B, N_OPROJ_TG) f32 — per-TG x2 sums from Dispatch 1
        n_inter: routed intermediate size
        k_hidden: hidden size K
        batch_size: B (1..8)
        n_active: experts per token (top_k)
        n_oproj_tg: number of o_proj TGs (for x2 partial sum reduction)
        n_experts: total number of experts E
        shared_inter: shared intermediate size (defaults to n_inter)
        group_size: quantization group size (default 64)

    Returns:
        (Y_routed, Y_shared, out_inds, norm_scores, gate_raw):
            Y_routed: (B * n_active, n_inter) f32
            Y_shared: (B, shared_inter) f32
            out_inds: (B * n_active,) uint32
            norm_scores: (B * n_active,) f32
            gate_raw: (B,) f32 — raw shared expert gate values (sigmoid in epilogue)
    """
    B = int(batch_size)
    n_inter_val = int(n_inter)
    shared_inter_val = int(shared_inter) if shared_inter is not None else n_inter_val
    k_val = int(k_hidden)
    n_active_val = int(n_active)
    top_k = n_active_val
    E = int(n_experts)
    n_oproj_tg_val = int(n_oproj_tg)

    kern = _get_batched_softmax_topk_swiglu_kernel(
        n_inter_val, shared_inter_val, k_val, n_active_val, B,
        E, top_k, True, int(group_size), n_oproj_tg_val,
    )

    max_inter = max(n_inter_val, shared_inter_val)
    total_routed = B * n_active_val

    results = kern(
        inputs=[
            w_gu, s_gu, b_gu,
            w_shared, s_shared, b_shared,
            w_seg, s_seg, b_seg,
            h_scaled,
            gate_part_a, gate_part_b,
            x2_partials,
        ],
        output_shapes=[
            (total_routed * n_inter_val,),        # Y_routed flat
            (B * shared_inter_val,),               # Y_shared flat
            (total_routed,),                       # out_inds
            (total_routed,),                       # norm_scores
            (B,),                                  # gate_raw
        ],
        output_dtypes=[
            mx.float32,    # Y_routed
            mx.float32,    # Y_shared
            mx.uint32,     # out_inds
            mx.float32,    # norm_scores
            mx.float32,    # gate_raw
        ],
        grid=(32, ceil_div(max_inter, 8) * 2, total_routed + 1),
        threadgroup=(32, 2, 1),
    )

    Y_routed = results[0].reshape(total_routed, n_inter_val)
    Y_shared = results[1].reshape(B, shared_inter_val)
    out_inds = results[2]
    norm_scores = results[3]
    gate_raw = results[4]

    return Y_routed, Y_shared, out_inds, norm_scores, gate_raw
