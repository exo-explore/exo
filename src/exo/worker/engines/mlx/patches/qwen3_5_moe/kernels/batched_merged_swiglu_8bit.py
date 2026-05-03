"""Batched merged 8-bit fused gate+up+SwiGLU for Qwen3.5 routed + shared experts.

Adapts merged_routed_shared_swiglu_8bit for batch_size B (1..8).

Grid z-dimension: B * n_active + 1
  - tgid.z < B * n_active: routed expert SwiGLU (one TG per batch×expert pair)
    batch_id = tgid.z / n_active, expert = inds[tgid.z]
    Input x indexed by batch_id * K
  - tgid.z == B * n_active: shared expert SwiGLU (ONE TG, register-level
    weight sharing — loads gate+up weights once, computes B outputs)

All constants baked into Metal source (no scalar kernel inputs).
"""
import mlx.core as mx


def ceil_div(a, b):
    return (a + b - 1) // b


def _gen_batched_swiglu_source(N_INTER, SHARED_INTER, K, n_active, B, group_size=64, with_seg=False, routed_only=False):
    gs = group_size
    sc_stride = 256 // gs
    slid_divisor = gs // 8
    N_TOTAL = 2 * N_INTER
    K_groups = K // gs
    SHARED_N_TOTAL = 2 * SHARED_INTER
    SHARED_K_groups = K // gs
    total_routed = B * n_active
    max_inter = max(N_INTER, SHARED_INTER)

    # Phase 3 (seg GEMV) — only emitted when with_seg=True
    if with_seg:
        seg_phase = f"""
    }} else if (tgid.z == (uint)(TOTAL_ROUTED + 1)) {{
        // ═══════ SEG (shared_expert_gate) PHASE 3 — sgid==0 only, register-shared across B ═══════
        const int VPT = 8;
        const int SEG_BLOCK = 256;
        const device uint8_t* seg_w_base = (const device uint8_t*)W_seg + slid * VPT;
        const device bfloat16_t* seg_sc_base = (const device bfloat16_t*)S_seg + slid / {slid_divisor};
        const device bfloat16_t* seg_bi_base = (const device bfloat16_t*)B_seg + slid / {slid_divisor};

        for (int b = 0; b < BATCH_SIZE; b++) {{
            const device uint8_t* seg_w_ptr = seg_w_base;
            const device bfloat16_t* seg_sc = seg_sc_base;
            const device bfloat16_t* seg_bi = seg_bi_base;
            int seg_xb = slid * VPT;
            float seg_acc = 0.0f;

            for (int k = 0; k < K_DIM; k += SEG_BLOCK) {{
                float seg_w_regs[VPT];
                for (int i = 0; i < VPT; i++) seg_w_regs[i] = float(seg_w_ptr[i]);
                float sc_val = float(*seg_sc);
                float bi_val = float(*seg_bi);

                float xsum = 0.0f, wacc = 0.0f;
                for (int i = 0; i < VPT; i++) {{
                    float xi = float(X[b * K_DIM + seg_xb + i]);
                    xsum += xi;
                    wacc += xi * seg_w_regs[i];
                }}
                seg_acc += sc_val * wacc + xsum * bi_val;

                seg_w_ptr += SEG_BLOCK;
                seg_sc += {sc_stride};
                seg_bi += {sc_stride};
                seg_xb += SEG_BLOCK;
            }}

            seg_acc = simd_sum(seg_acc);
            if (slid == 0) {{
                gate_raw[b] = seg_acc;
            }}
        }}"""
    else:
        seg_phase = ""

    # Shared expert: generate unrolled batch loops for gate+up
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

    # Generate base source
    base_src = f"""
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

    uint3 tgid = threadgroup_position_in_grid;
    uint sgid = simdgroup_index_in_threadgroup;
    uint slid = thread_index_in_simdgroup;

    int out_row = tgid.y * 8 + sgid * RESULTS_PER_SG;

    int row_limit;
    if (tgid.z < (uint)TOTAL_ROUTED) row_limit = N_INTER_DIM;
    else if (tgid.z == (uint)TOTAL_ROUTED) row_limit = SHARED_INTER_DIM;
    else row_limit = 1;  // seg (Phase 3) — only out_row==0 survives
    if (out_row >= row_limit) return;

    if (tgid.z < (uint)TOTAL_ROUTED) {{
        // ═══════ ROUTED EXPERT PATH ═══════
        int flat_idx = (int)tgid.z;
        int batch_id = flat_idx / N_ACTIVE;
        int expert = inds[flat_idx];

        // Gate weights: first N_INTER rows of expert's stacked weight
        const device uint8_t* ws_gate = (const device uint8_t*)W
            + (long)expert * N_TOTAL * K_DIM + out_row * K_DIM + slid * VALUES_PER_THREAD;
        const device bfloat16_t* sc_gate = (const device bfloat16_t*)S
            + (long)expert * N_TOTAL * K_GROUPS + out_row * K_GROUPS + slid / {slid_divisor};
        const device bfloat16_t* bi_gate = (const device bfloat16_t*)B_q
            + (long)expert * N_TOTAL * K_GROUPS + out_row * K_GROUPS + slid / {slid_divisor};

        // Up weights: second N_INTER rows
        const device uint8_t* ws_up = (const device uint8_t*)W
            + (long)expert * N_TOTAL * K_DIM + (out_row + N_INTER_DIM) * K_DIM + slid * VALUES_PER_THREAD;
        const device bfloat16_t* sc_up = (const device bfloat16_t*)S
            + (long)expert * N_TOTAL * K_GROUPS + (out_row + N_INTER_DIM) * K_GROUPS + slid / {slid_divisor};
        const device bfloat16_t* bi_up = (const device bfloat16_t*)B_q
            + (long)expert * N_TOTAL * K_GROUPS + (out_row + N_INTER_DIM) * K_GROUPS + slid / {slid_divisor};

        // Input: x for this batch element
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

        device float* yp = Y_routed + flat_idx * N_INTER_DIM + out_row;
        for (int row = 0; row < RESULTS_PER_SG; row++) {{
            float g = simd_sum(gate_result[row]);
            float u = simd_sum(up_result[row]);
            if (slid == 0) {{
                float silu_g = g / (1.0f + metal::exp(-g));
                yp[row] = silu_g * u;
            }}
        }}

    }} else if (tgid.z == (uint)TOTAL_ROUTED) {{
        // ═══════ SHARED EXPERT PATH (register-level weight sharing) ═══════
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

        // SwiGLU epilogue + write for all {B} batches
{shared_write}{seg_phase}
    }}
"""
    if routed_only:
        # Replace the shared expert branch body with an early return.
        # Anchor the start of the shared branch.
        shared_start = "} else if (tgid.z == (uint)TOTAL_ROUTED) {"
        if shared_start in base_src:
            # Find the next `}} else if` (seg branch) OR the closing of the function block.
            after = base_src.split(shared_start, 1)[1]
            seg_marker = "} else if (tgid.z == (uint)(TOTAL_ROUTED + 1)) {"
            if seg_marker in after:
                # Replace shared body up to seg marker with early-return
                shared_body, rest = after.split(seg_marker, 1)
                replacement = "        return;\n    " + seg_marker + rest
                base_src = base_src.split(shared_start, 1)[0] + shared_start + "\n        " + replacement
            else:
                # No seg branch; shared branch ends at function-close. Find closing.
                # Just replace shared_start with shared_start + return.
                base_src = base_src.replace(shared_start, shared_start + "\n        return;\n        // (shared body skipped due to routed_only)\n        if(0){{", 1) + "}}"
                # ^ adds a sentinel if(0) { ... } that swallows the original shared body
                # then closes the if block to keep braces balanced
    return base_src


_batched_swiglu_cache = {}


def _get_batched_swiglu_kernel(N_INTER, SHARED_INTER, K, n_active, B, group_size=64):
    key = (N_INTER, SHARED_INTER, K, n_active, B, group_size)
    if key not in _batched_swiglu_cache:
        _batched_swiglu_cache[key] = mx.fast.metal_kernel(
            name=f"batched_swiglu_NI{N_INTER}_SI{SHARED_INTER}_K{K}_na{n_active}_B{B}",
            input_names=["W", "S", "B_q", "W_shared", "S_shared", "B_shared",
                         "X", "inds"],
            output_names=["Y_routed", "Y_shared"],
            source=_gen_batched_swiglu_source(N_INTER, SHARED_INTER, K, n_active, B, group_size),
        )
    return _batched_swiglu_cache[key]


def batched_merged_swiglu_8bit(w_q, s, b_q,
                                w_shared, s_shared, b_shared,
                                x, inds,
                                n_inter, k_hidden, batch_size,
                                n_active, group_size=64,
                                shared_inter=None):
    """Batched merged gate+up+SwiGLU for 8-bit routed + shared experts.

    Args:
        w_q: stacked routed weights (E, 2*N_INTER, K/4) uint32
        s: routed scales (E, 2*N_INTER, K/gs) bf16
        b_q: routed biases (E, 2*N_INTER, K/gs) bf16
        w_shared: shared gate+up stacked (2*SHARED_INTER, K/4) uint32
        s_shared: shared scales (2*SHARED_INTER, K/gs) bf16
        b_shared: shared biases (2*SHARED_INTER, K/gs) bf16
        x: (B, K) bf16 — input vectors
        inds: (B * n_active,) uint32 — flattened expert indices
        n_inter: routed intermediate size
        k_hidden: hidden size
        batch_size: B
        n_active: experts per token (top_k)
        shared_inter: shared intermediate size (defaults to n_inter)

    Returns:
        Y_routed: (B * n_active, n_inter) f32
        Y_shared: (B, shared_inter) f32
    """
    B = batch_size
    n_inter_val = int(n_inter)
    shared_inter_val = int(shared_inter) if shared_inter is not None else n_inter_val
    k_val = int(k_hidden)

    kern = _get_batched_swiglu_kernel(n_inter_val, shared_inter_val, k_val, n_active, B)

    max_inter = max(n_inter_val, shared_inter_val)
    total_routed = B * n_active

    Y = kern(
        inputs=[w_q, s, b_q, w_shared, s_shared, b_shared,
                x, inds],
        output_shapes=[(total_routed * n_inter_val,), (B * shared_inter_val,)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(32, ceil_div(max_inter, 8) * 2, total_routed + 1),
        threadgroup=(32, 2, 1),
    )

    return Y[0].reshape(total_routed, n_inter_val), Y[1].reshape(B, shared_inter_val)


_batched_swiglu_seg_cache = {}


def _get_batched_swiglu_with_seg_kernel(N_INTER, SHARED_INTER, K, n_active, B, group_size=64):
    key = (N_INTER, SHARED_INTER, K, n_active, B, group_size)
    if key not in _batched_swiglu_seg_cache:
        _batched_swiglu_seg_cache[key] = mx.fast.metal_kernel(
            name=f"batched_swiglu_seg_NI{N_INTER}_SI{SHARED_INTER}_K{K}_na{n_active}_B{B}",
            input_names=["W", "S", "B_q", "W_shared", "S_shared", "B_shared",
                         "X", "inds", "W_seg", "S_seg", "B_seg"],
            output_names=["Y_routed", "Y_shared", "gate_raw"],
            source=_gen_batched_swiglu_source(N_INTER, SHARED_INTER, K, n_active, B, group_size, with_seg=True),
        )
    return _batched_swiglu_seg_cache[key]


def batched_merged_swiglu_with_seg_8bit(w_q, s, b_q,
                                          w_shared, s_shared, b_shared,
                                          x, inds,
                                          w_seg, s_seg, b_seg,
                                          n_inter, k_hidden, batch_size,
                                          n_active, group_size=64,
                                          shared_inter=None):
    """Like batched_merged_swiglu_8bit but also fuses shared_expert_gate (1, K) GEMV
    into Phase 3 of the kernel — emits gate_raw[B] (raw scalar per batch element)
    as a third output, eliminating the separate stock matmul dispatch.

    Extra args:
        w_seg: shared_expert_gate.weight (1, K/4) uint32
        s_seg: shared_expert_gate.scales (1, K/gs) bf16
        b_seg: shared_expert_gate.biases (1, K/gs) bf16

    Extra return:
        gate_raw: (B,) f32 — raw shared_expert_gate output per batch element.
                  (Sigmoid is applied later inside batched_moe_epilogue.)
    """
    B = batch_size
    n_inter_val = int(n_inter)
    shared_inter_val = int(shared_inter) if shared_inter is not None else n_inter_val
    k_val = int(k_hidden)
    n_active_val = int(n_active)
    max_inter = max(n_inter_val, shared_inter_val)

    kern = _get_batched_swiglu_with_seg_kernel(n_inter_val, shared_inter_val, k_val, n_active_val, B, group_size)

    grid_y = ceil_div(max_inter, 8) * 2  # match original: 2 sgids per TG → grid_y is 2× the row count
    grid_z = B * n_active_val + 2  # +1 for shared expert, +1 for seg phase

    results = kern(
        inputs=[w_q, s, b_q, w_shared, s_shared, b_shared, x, inds,
                w_seg, s_seg, b_seg],
        output_shapes=[(B * n_active_val * n_inter_val,), (B * shared_inter_val,), (B,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
        grid=(32, grid_y, grid_z),
        threadgroup=(32, 2, 1),  # 2 sgids per TG (matches kernel's out_row formula)
    )
    return results[0].reshape(B * n_active_val, n_inter_val), results[1].reshape(B, shared_inter_val), results[2]


_batched_swiglu_routed_seg_cache = {}


def _get_batched_swiglu_routed_with_seg_kernel(N_INTER, SHARED_INTER, K, n_active, B, group_size=64):
    key = (N_INTER, SHARED_INTER, K, n_active, B, group_size)
    if key not in _batched_swiglu_routed_seg_cache:
        _batched_swiglu_routed_seg_cache[key] = mx.fast.metal_kernel(
            name=f"batched_swiglu_routedonly_seg_NI{N_INTER}_SI{SHARED_INTER}_K{K}_na{n_active}_B{B}",
            input_names=["W", "S", "B_q", "W_shared", "S_shared", "B_shared",
                         "X", "inds", "W_seg", "S_seg", "B_seg"],
            output_names=["Y_routed", "Y_shared", "gate_raw"],
            source=_gen_batched_swiglu_source(N_INTER, SHARED_INTER, K, n_active, B,
                                              group_size, with_seg=True, routed_only=True),
        )
    return _batched_swiglu_routed_seg_cache[key]


def batched_merged_routed_swiglu_with_seg_8bit(w_q, s, b_q,
                                                  w_shared, s_shared, b_shared,
                                                  x, inds,
                                                  w_seg, s_seg, b_seg,
                                                  n_inter, k_hidden, batch_size,
                                                  n_active, group_size=64,
                                                  shared_inter=None):
    """Routed-only SwiGLU + seg (no shared SwiGLU phase).

    Same kernel signature as batched_merged_swiglu_with_seg_8bit but the shared
    expert branch is an early-return — shared SwiGLU is computed elsewhere
    (e.g., shared_swiglu_with_routing_8bit). Y_shared output buffer is allocated
    but not written; ignored on return.

    Returns: (y_routed, gate_raw)
    """
    B = batch_size
    n_inter_val = int(n_inter)
    shared_inter_val = int(shared_inter) if shared_inter is not None else n_inter_val
    k_val = int(k_hidden)
    n_active_val = int(n_active)
    max_inter = max(n_inter_val, shared_inter_val)

    kern = _get_batched_swiglu_routed_with_seg_kernel(n_inter_val, shared_inter_val, k_val, n_active_val, B, group_size)

    grid_y = ceil_div(max_inter, 8) * 2
    grid_z = B * n_active_val + 2  # still +1 for shared (no-op) and +1 for seg

    results = kern(
        inputs=[w_q, s, b_q, w_shared, s_shared, b_shared, x, inds,
                w_seg, s_seg, b_seg],
        output_shapes=[(B * n_active_val * n_inter_val,), (B * shared_inter_val,), (B,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
        grid=(32, grid_y, grid_z),
        threadgroup=(32, 2, 1),
    )
    return results[0].reshape(B * n_active_val, n_inter_val), results[2]

