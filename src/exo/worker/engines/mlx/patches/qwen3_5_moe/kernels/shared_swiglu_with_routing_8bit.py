"""Heterogeneous-grid kernel: shared expert SwiGLU || (softmax + topk + normalize).

Lifts the shared expert SwiGLU pattern from `batched_merged_swiglu_8bit.py`
(Z=0, multi-TG, register-sharing across B) and the routing pipeline from
`batched_softmax_topk_swiglu_8bit.py` (Z=1, single TG, sequential
softmax→argpartition→take_along_axis→normalize in TG memory) into one
kernel. Both Z slots run concurrently — outputs are disjoint.

Replaces these stock + custom dispatches:
  - mx.softmax(gate_scores, precise=True)       (1 TG)
  - mx.argpartition(softmax_out, kth=-k)        (1 TG)
  - mx.take_along_axis(softmax_out, inds)       (~tiny)
  - scores / scores.sum()                        (~tiny)
  - shared SwiGLU phase of batched_merged_swiglu (multi-TG)
→ ONE dispatch.

Inputs:
  W_shared (2*SHARED_INTER, K/4) uint32        — shared expert gate+up packed
  S_shared (2*SHARED_INTER, K/gs) bf16         — shared expert scales
  B_shared (2*SHARED_INTER, K/gs) bf16         — shared expert biases
  gate_scores (B, E) f32                        — output of stock moe.gate(x)
  X (B, K) bf16                                 — post-attention-LN hidden
Outputs:
  Y_shared (B, SHARED_INTER) f32                — shared SwiGLU output
  inds (B, k) uint32                            — top-k expert indices
  norm_scores (B, k) f32                        — top-k scores (normalized if norm_topk)
"""
import mlx.core as mx


def ceil_div(a, b):
    return (a + b - 1) // b


def _gen_source(SHARED_INTER, K, B, E, K_TOP, group_size=64, norm_topk=True):
    gs = group_size
    sc_stride = 256 // gs
    slid_divisor = gs // 8
    SHARED_N_TOTAL = 2 * SHARED_INTER
    SHARED_K_groups = K // gs
    # SPT: scores per thread for routing (E split across 64 threads)
    SPT = ceil_div(E, 64)

    # Shared expert: register-sharing across B (lifted from batched_merged_swiglu_8bit.py)
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

    shared_write = "\n".join(f"""
        for (int row = 0; row < RESULTS_PER_SG; row++) {{
            float g{b} = simd_sum(gate{b}[row]);
            float u{b} = simd_sum(up{b}[row]);
            if (slid == 0) {{
                float silu_g{b} = g{b} / (1.0f + metal::exp(-g{b}));
                Y_shared[{b} * SHARED_INTER_DIM + out_row + row] = silu_g{b} * u{b};
            }}
        }}""" for b in range(B))

    norm_factor = "1.0f / total_score" if norm_topk else "1.0f"

    return f"""
    const int RESULTS_PER_SG = 4;
    const int VALUES_PER_THREAD = 8;
    const int BLOCK_SIZE = 256;
    const int SHARED_INTER_DIM = {SHARED_INTER};
    const int K_DIM = {K};
    const int K_GROUPS = {SHARED_K_groups};
    const int E_CONST = {E};
    const int K_TOP_CONST = {K_TOP};
    const int BATCH_SIZE = {B};
    const int SPT = {SPT};

    uint3 tgid = threadgroup_position_in_grid;
    uint sgid = simdgroup_index_in_threadgroup;
    uint slid = thread_index_in_simdgroup;
    int tid = int(sgid) * 32 + int(slid);

    // Each TG covers RESULTS_PER_SG rows per simdgroup × 2 sgids = 8 rows
    int out_row = tgid.y * 8 + sgid * RESULTS_PER_SG;

    if (tgid.z == 0) {{
        // ═══════════ Z=0: SHARED EXPERT SWIGLU (multi-TG, register-shared across B) ═══════════
        if (out_row >= SHARED_INTER_DIM) return;

        const device uint8_t* ws_gate = (const device uint8_t*)W_shared
            + (long)out_row * K_DIM + slid * VALUES_PER_THREAD;
        const device bfloat16_t* sc_gate = (const device bfloat16_t*)S_shared
            + (long)out_row * K_GROUPS + slid / {slid_divisor};
        const device bfloat16_t* bi_gate = (const device bfloat16_t*)B_shared
            + (long)out_row * K_GROUPS + slid / {slid_divisor};

        const device uint8_t* ws_up = (const device uint8_t*)W_shared
            + (long)(out_row + SHARED_INTER_DIM) * K_DIM + slid * VALUES_PER_THREAD;
        const device bfloat16_t* sc_up = (const device bfloat16_t*)S_shared
            + (long)(out_row + SHARED_INTER_DIM) * K_GROUPS + slid / {slid_divisor};
        const device bfloat16_t* bi_up = (const device bfloat16_t*)B_shared
            + (long)(out_row + SHARED_INTER_DIM) * K_GROUPS + slid / {slid_divisor};

        int x_base = slid * VALUES_PER_THREAD;
        {shared_result_decls}

        for (int k = 0; k < K_DIM; k += BLOCK_SIZE) {{
{shared_x_load}

            for (int row = 0; row < RESULTS_PER_SG; row++) {{
                const device uint8_t* wg = ws_gate + row * K_DIM;
                float sg = float(sc_gate[row * K_GROUPS]);
                float bg = float(bi_gate[row * K_GROUPS]);
                float wg_vals[VALUES_PER_THREAD];
                for (int i = 0; i < VALUES_PER_THREAD; i++) wg_vals[i] = float(wg[i]);
{shared_gate_qdot}

                const device uint8_t* wu = ws_up + row * K_DIM;
                float su = float(sc_up[row * K_GROUPS]);
                float bu = float(bi_up[row * K_GROUPS]);
                float wu_vals[VALUES_PER_THREAD];
                for (int i = 0; i < VALUES_PER_THREAD; i++) wu_vals[i] = float(wu[i]);
{shared_up_qdot}
            }}

            ws_gate += BLOCK_SIZE; ws_up += BLOCK_SIZE;
            sc_gate += {sc_stride}; sc_up += {sc_stride};
            bi_gate += {sc_stride}; bi_up += {sc_stride};
            x_base += BLOCK_SIZE;
        }}

{shared_write}

    }} else {{
        // ═══════════ Z=1: ROUTING (single TG, sequential softmax → topk → normalize) ═══════════
        if (tgid.y > 0) return;  // only one TG does routing

        threadgroup int tg_inds[K_TOP_CONST];
        threadgroup float tg_selected_scores[K_TOP_CONST];
        threadgroup float tg_softmax_sg[2];
        threadgroup float tg_tk_val[2];
        threadgroup int tg_tk_info[2];

        for (int batch_id = 0; batch_id < BATCH_SIZE; batch_id++) {{
            // Phase A: load gate_scores (B, E) into per-thread registers
            float my_scores[SPT];
            for (int j = 0; j < SPT; j++) {{
                int e = tid * SPT + j;
                my_scores[j] = (e < E_CONST) ? gate_scores[batch_id * E_CONST + e] : -1e30f;
            }}

            // Phase B: softmax (max-reduce, exp, sum-reduce, normalize)
            float local_max = -1e30f;
            for (int j = 0; j < SPT; j++) local_max = max(local_max, my_scores[j]);
            float sg_max_val = simd_max(local_max);
            if (slid == 0) tg_softmax_sg[sgid] = sg_max_val;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float tg_max = max(tg_softmax_sg[0], tg_softmax_sg[1]);

            float local_sum = 0.0f;
            for (int j = 0; j < SPT; j++) {{
                float ev = metal::exp(my_scores[j] - tg_max);
                my_scores[j] = ev;
                local_sum += ev;
            }}
            float sg_sum_val = simd_sum(local_sum);
            if (slid == 0) tg_softmax_sg[sgid] = sg_sum_val;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float tg_sum = tg_softmax_sg[0] + tg_softmax_sg[1];

            float inv_sum = 1.0f / tg_sum;
            for (int j = 0; j < SPT; j++) my_scores[j] *= inv_sum;

            // Phase C: parallel top-k via select-max-then-mask (K_TOP rounds)
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

            // Phase D: write inds + (optionally normalized) scores to global
            if (tid == 0) {{
                float total_score = 0.0f;
                for (int a = 0; a < K_TOP_CONST; a++) total_score += tg_selected_scores[a];
                float inv_total = {norm_factor};
                for (int a = 0; a < K_TOP_CONST; a++) {{
                    inds[batch_id * K_TOP_CONST + a] = (uint)tg_inds[a];
                    norm_scores[batch_id * K_TOP_CONST + a] = tg_selected_scores[a] * inv_total;
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
    }}
"""


_kernel_cache = {}


def _get_kernel(SHARED_INTER, K, B, E, K_TOP, group_size=64, norm_topk=True):
    key = (SHARED_INTER, K, B, E, K_TOP, group_size, norm_topk)
    if key not in _kernel_cache:
        nt = "T" if norm_topk else "F"
        _kernel_cache[key] = mx.fast.metal_kernel(
            name=f"shared_swiglu_routing_SI{SHARED_INTER}_K{K}_B{B}_E{E}_KT{K_TOP}_{nt}",
            input_names=["W_shared", "S_shared", "B_shared", "gate_scores", "X"],
            output_names=["Y_shared", "inds", "norm_scores"],
            source=_gen_source(SHARED_INTER, K, B, E, K_TOP, group_size, norm_topk),
        )
    return _kernel_cache[key]


def shared_swiglu_with_routing_8bit(w_shared, s_shared, b_shared,
                                     gate_scores, x,
                                     k_hidden, batch_size,
                                     n_active, n_experts,
                                     shared_inter, group_size=64,
                                     norm_topk_prob=True):
    """Run shared SwiGLU and routing in one heterogeneous-grid dispatch.

    Args:
        w_shared, s_shared, b_shared: shared expert gate+up stacked weights/scales/biases
        gate_scores: (B, E) f32 — pre-softmax scores from moe.gate(x)
        x: (B, K) bf16 — post-attention-layernorm hidden state
        k_hidden, batch_size, n_active, n_experts, shared_inter: shape constants
        norm_topk_prob: True to normalize top-k scores by their sum (matches mlx-lm behavior)

    Returns:
        y_shared: (B, SHARED_INTER) f32
        inds: (B, n_active) uint32
        norm_scores: (B, n_active) f32
    """
    B = batch_size
    SHARED_INTER = int(shared_inter)
    K = int(k_hidden)
    E = int(n_experts)
    K_TOP = int(n_active)

    kern = _get_kernel(SHARED_INTER, K, B, E, K_TOP, group_size, bool(norm_topk_prob))

    grid_y = ceil_div(SHARED_INTER, 8) * 2  # 2 sgids/TG → grid_y is 2× row-tile count
    grid_z = 2  # Z=0: shared SwiGLU; Z=1: routing

    Y_shared, inds, norm_scores = kern(
        inputs=[w_shared, s_shared, b_shared, gate_scores, x],
        output_shapes=[(B * SHARED_INTER,), (B * K_TOP,), (B * K_TOP,)],
        output_dtypes=[mx.float32, mx.uint32, mx.float32],
        grid=(32, grid_y, grid_z),
        threadgroup=(32, 2, 1),  # 64 threads/TG = 2 sgids
    )
    return Y_shared.reshape(B, SHARED_INTER), inds.reshape(B, K_TOP), norm_scores.reshape(B, K_TOP)
