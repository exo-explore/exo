"""Custom SDPA Pass 2 + gate multiply for GQA attention (Dispatch 5).

Combines the block-partial results from Pass 1 into final SDPA output,
with gate multiply epilogue: output *= gate_sigmoid.

Modeled on MLX's sdpa_vector_2pass_2 kernel with two improvements:
  1. V-dimension splitting (V_SPLIT=4): Instead of 1 TG per head (16 TGs,
     20% M3 Ultra utilization), split the 256-dim output across 4 TGs per
     head -> 64 TGs (80% utilization).
  2. Gate multiply epilogue: Apply output *= gate_sigmoid in the same kernel.

Each TG = 1024 threads (32 SGs × 32 threads), matching MLX's structure:
  - Handles D/V_SPLIT = 64 V-dimensions per TG
  - 32 SGs process 32 blocks in parallel per iteration
  - blocks/32 = 4 iterations to cover all 128 blocks
  - Shared memory transpose + simd_sum to reduce across SGs
  - Each thread handles EPT = 2 V-elements

Grid: (H_q * V_SPLIT * 32, 32, B)
TG:   (32, 32, 1)
"""

import mlx.core as mx


def _gen_sdpa_pass2_gate_source(D=256, V_SPLIT=4):
    """Generate Metal source for SDPA Pass 2 + gate multiply.

    Modeled on MLX's sdpa_vector_2pass_2 but with V_SPLIT for higher
    TG count and fused gate multiply epilogue.

    Inputs:
      o_partials:   [B, H_q, blocks, D] bf16
      sums:         [B, H_q, blocks] f32
      maxs:         [B, H_q, blocks] f32
      gate_sigmoid: [B, 1, H_q * D] f32

    Output:
      attn_output:  [B, 1, H_q * D] bf16

    TG assignment:
      tgid.x = head_idx * V_SPLIT + split_idx
      Each TG handles D/V_SPLIT V-elements for one head.
      32 SGs within TG process blocks in parallel.
    """
    BN = 32   # number of SGs per TG = blocks processed in parallel
    V_PER_SPLIT = D // V_SPLIT      # 64
    EPT = V_PER_SPLIT // BN          # 2 elements per thread

    return f"""
    const int D_DIM = {D};
    const int V_SPLIT = {V_SPLIT};
    const int V_PER_SPLIT = {V_PER_SPLIT};
    const int EPT = {EPT};
    const int BN = {BN};

    uint tg_idx = threadgroup_position_in_grid.x;
    uint simd_gid = simdgroup_index_in_threadgroup;  // 0..31 (which SG = block offset)
    uint simd_lid = thread_index_in_simdgroup;        // 0..31 (thread within SG)
    uint b_idx = threadgroup_position_in_grid.z;

    int head_idx = (int)tg_idx / V_SPLIT;
    int split_idx = (int)tg_idx % V_SPLIT;
    int v_offset = split_idx * V_PER_SPLIT;

    int blocks_val = (int)N_blocks;
    int h_q_val = (int)H_Q;

    // Base offset into partials/maxs/sums for this head: [B, H_q, blocks, ...]
    int head_base = ((int)b_idx * h_q_val + head_idx);

    // Pointer setup for partials: [B, H_q, blocks, D]
    // Each SG starts at a different block, reads EPT V-elements at v_offset
    int p_base = head_base * blocks_val * D_DIM
               + (int)simd_gid * D_DIM
               + v_offset + (int)simd_lid * EPT;
    int ms_base = head_base * blocks_val;

    // ── Phase 1: Find global max across all blocks ──
    // Each of 32 simd_lids processes blocks/32 = 4 blocks (strided)
    float local_max = -__FLT_MAX__;
    for (int b = 0; b < blocks_val / BN; ++b) {{
        local_max = metal::max(local_max, maxs[ms_base + (int)simd_lid + BN * b]);
    }}
    float global_max = simd_max(local_max);

    // ── Phase 2: Compute global sum_exp ──
    float local_sum = 0.0f;
    for (int b = 0; b < blocks_val / BN; ++b) {{
        float factor = metal::fast::exp(maxs[ms_base + (int)simd_lid + BN * b] - global_max);
        local_sum += factor * sums[ms_base + (int)simd_lid + BN * b];
    }}
    float global_sum = simd_sum(local_sum);

    // ── Phase 3: Accumulate V-partials (block-parallel via SGs) ──
    // 32 SGs process 32 blocks simultaneously, 4 iterations for 128 blocks.
    // Each SG reads EPT V-elements from its assigned block.
    float o[EPT] = {{0}};
    for (int b = 0; b < blocks_val / BN; ++b) {{
        float factor = metal::fast::exp(maxs[ms_base + (int)simd_gid] - global_max);
        for (int i = 0; i < EPT; i++) {{
            o[i] += factor * (float)o_partials[p_base + i];
        }}
        // Advance maxs by BN and partials by BN * D
        ms_base += BN;
        p_base += BN * D_DIM;
    }}

    // ── Phase 4: Shared memory transpose + reduce across SGs ──
    // Each SG has accumulated partials for different blocks but same V-elements.
    // Transpose so each SG's threads hold all block contributions for the same V position,
    // then simd_sum reduces them.
    threadgroup float tg_mem[BN * BN];  // 32×32 = 1024 floats = 4 KB

    for (int i = 0; i < EPT; i++) {{
        tg_mem[(int)simd_lid * BN + (int)simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(tg_mem[(int)simd_gid * BN + (int)simd_lid]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // ── Phase 5: Normalize + gate multiply + write (simd_lid == 0 only) ──
    if (simd_lid == 0) {{
        float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
        int out_base = (int)b_idx * h_q_val * D_DIM + head_idx * D_DIM
                     + v_offset + (int)simd_gid * EPT;
        for (int i = 0; i < EPT; i++) {{
            float val = o[i] * inv_sum;
            float gate = gate_sigmoid[out_base + i];
            attn_output[out_base + i] = static_cast<bfloat16_t>(val * gate);
        }}
    }}
"""


_pass2_cache = {}


def _get_pass2_kernel(D, V_SPLIT):
    """Get or compile the SDPA Pass 2 + gate kernel."""
    key = (D, V_SPLIT)
    if key not in _pass2_cache:
        _pass2_cache[key] = mx.fast.metal_kernel(
            name="custom_sdpa_pass2_gate",
            input_names=["o_partials", "sums", "maxs", "gate_sigmoid",
                         "H_Q", "N_blocks"],
            output_names=["attn_output"],
            source=_gen_sdpa_pass2_gate_source(D, V_SPLIT),
        )
    return _pass2_cache[key]


def custom_sdpa_pass2_gate(o_partials, sums, maxs, gate_sigmoid,
                           H_q, D, blocks=128, V_SPLIT=4, batch_size=1,
                           scalars=None):
    """SDPA Pass 2: reduce block partials + gate multiply.

    Args:
        o_partials:   [B, H_q, blocks, D] bf16 — from Pass 1.
        sums:         [B, H_q, blocks] f32 — from Pass 1.
        maxs:         [B, H_q, blocks] f32 — from Pass 1.
        gate_sigmoid: [B, 1, H_q*D] f32 — from Dispatch 1.
        H_q: int — number of query heads.
        D: int — head dimension.
        blocks: int — number of blocks from Pass 1.
        V_SPLIT: int — V-dimension split factor (4 for 80% M3 Ultra util).
        batch_size: int
        scalars: dict of pre-cached mx.array scalars (optional)

    Returns:
        attn_output: [B, 1, H_q*D] bf16 — final gated attention output.
    """
    B = batch_size

    kern = _get_pass2_kernel(D, V_SPLIT)

    if scalars is not None:
        h_q_arr = scalars['H_Q']
        n_blocks_arr = scalars['N_blocks']
    else:
        h_q_arr = mx.array(H_q, dtype=mx.int32)
        n_blocks_arr = mx.array(blocks, dtype=mx.int32)

    # Flatten inputs for kernel
    o_flat = o_partials.reshape(B * H_q * blocks * D)
    s_flat = sums.reshape(B * H_q * blocks)
    m_flat = maxs.reshape(B * H_q * blocks)
    g_flat = gate_sigmoid.reshape(B * H_q * D)

    n_tgs = H_q * V_SPLIT
    BN = 32  # SGs per TG
    results = kern(
        inputs=[o_flat, s_flat, m_flat, g_flat, h_q_arr, n_blocks_arr],
        output_shapes=[(B * H_q * D,)],
        output_dtypes=[mx.bfloat16],
        grid=(n_tgs * BN, BN, B),
        threadgroup=(BN, BN, 1),
    )

    return results[0].reshape(B, 1, H_q * D)
