"""Custom SDPA Pass 2 + gate multiply for GQA attention (Dispatch 5).

32-SG block-parallel reduction + V_SPLIT for high GPU utilization.
All constants baked into Metal source (no scalar kernel inputs).
"""

import mlx.core as mx


def _gen_sdpa_pass2_gate_source(D=256, V_SPLIT=4, H_q=16, blocks=128):
    BN = 32
    V_PER_SPLIT = D // V_SPLIT
    EPT = V_PER_SPLIT // BN

    return f"""
    const int D_DIM = {D};
    const int V_SPLIT = {V_SPLIT};
    const int V_PER_SPLIT = {V_PER_SPLIT};
    const int EPT = {EPT};
    const int BN = {BN};
    const int H_Q = {H_q};
    const int N_BLOCKS = {blocks};

    uint tg_idx = threadgroup_position_in_grid.x;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    uint b_idx = threadgroup_position_in_grid.z;

    int head_idx = (int)tg_idx / V_SPLIT;
    int split_idx = (int)tg_idx % V_SPLIT;
    int v_offset = split_idx * V_PER_SPLIT;

    int head_base = ((int)b_idx * H_Q + head_idx);

    int p_base = head_base * N_BLOCKS * D_DIM
               + (int)simd_gid * D_DIM
               + v_offset + (int)simd_lid * EPT;
    int ms_base = head_base * N_BLOCKS;

    // ── Phase 1: Find global max across all blocks ──
    float local_max = -__FLT_MAX__;
    for (int b = 0; b < N_BLOCKS / BN; ++b) {{
        local_max = metal::max(local_max, maxs[ms_base + (int)simd_lid + BN * b]);
    }}
    float global_max = simd_max(local_max);

    // ── Phase 2: Compute global sum_exp ──
    float local_sum = 0.0f;
    for (int b = 0; b < N_BLOCKS / BN; ++b) {{
        float factor = metal::fast::exp(maxs[ms_base + (int)simd_lid + BN * b] - global_max);
        local_sum += factor * sums[ms_base + (int)simd_lid + BN * b];
    }}
    float global_sum = simd_sum(local_sum);

    // ── Phase 3: Accumulate V-partials (block-parallel via SGs) ──
    float o[EPT] = {{0}};
    for (int b = 0; b < N_BLOCKS / BN; ++b) {{
        float factor = metal::fast::exp(maxs[ms_base + (int)simd_gid] - global_max);
        for (int i = 0; i < EPT; i++) {{
            o[i] += factor * (float)o_partials[p_base + i];
        }}
        ms_base += BN;
        p_base += BN * D_DIM;
    }}

    // ── Phase 4: Shared memory transpose + reduce across SGs ──
    threadgroup float tg_mem[BN * BN];

    for (int i = 0; i < EPT; i++) {{
        tg_mem[(int)simd_lid * BN + (int)simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(tg_mem[(int)simd_gid * BN + (int)simd_lid]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // ── Phase 5: Normalize + gate multiply + write (simd_lid == 0 only) ──
    if (simd_lid == 0) {{
        float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
        int out_base = (int)b_idx * H_Q * D_DIM + head_idx * D_DIM
                     + v_offset + (int)simd_gid * EPT;
        for (int i = 0; i < EPT; i++) {{
            float val = o[i] * inv_sum;
            float gate = gate_sigmoid[out_base + i];
            attn_output[out_base + i] = static_cast<bfloat16_t>(val * gate);
        }}
    }}
"""


_pass2_cache = {}


def _get_pass2_kernel(D, V_SPLIT, H_q, blocks):
    key = (D, V_SPLIT, H_q, blocks)
    if key not in _pass2_cache:
        _pass2_cache[key] = mx.fast.metal_kernel(
            name=f"sdpa_pass2_gate_D{D}_vs{V_SPLIT}_Hq{H_q}_b{blocks}",
            input_names=["o_partials", "sums", "maxs", "gate_sigmoid"],
            output_names=["attn_output"],
            source=_gen_sdpa_pass2_gate_source(D, V_SPLIT, H_q, blocks),
        )
    return _pass2_cache[key]


def custom_sdpa_pass2_gate(o_partials, sums, maxs, gate_sigmoid,
                           H_q, D, blocks=128, V_SPLIT=4, batch_size=1,
                           scalars=None):
    B = batch_size

    kern = _get_pass2_kernel(D, V_SPLIT, H_q, blocks)

    o_flat = o_partials.reshape(B * H_q * blocks * D)
    s_flat = sums.reshape(B * H_q * blocks)
    m_flat = maxs.reshape(B * H_q * blocks)
    g_flat = gate_sigmoid.reshape(B * H_q * D)

    n_tgs = H_q * V_SPLIT
    BN = 32
    results = kern(
        inputs=[o_flat, s_flat, m_flat, g_flat],
        output_shapes=[(B * H_q * D,)],
        output_dtypes=[mx.bfloat16],
        grid=(n_tgs * BN, BN, B),
        threadgroup=(BN, BN, 1),
    )

    return results[0].reshape(B, 1, H_q * D)
