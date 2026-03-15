"""Custom SDPA Pass 1 for GQA attention (Dispatch 4).

Replicates MLX's sdpa_vector_2pass_1 kernel for decode (query_len=1).
Online softmax over N key positions, strided by `blocks`.

Each TG = (32, gqa_factor, 1):
  - 32 threads in x = 1 SIMD group, each handles D/32 = 8 Q/K/V elements
  - gqa_factor threads in y = one per Q head sharing the same KV head
  - Grid z = blocks (each block handles a stride of the key sequence)

Algorithm per block:
  1. Load query (once), scale by 1/sqrt(D)
  2. For each key position i in [block_idx, N) with stride `blocks`:
     a. Dot product: score = sum(q[j] * k[i,j]) via simd_sum
     b. Online softmax: update max, sum_exp, rescale accumulators
     c. Accumulate: o[j] += exp_score * v[i,j]
  3. Write partials: o_partials[block, head, D], sums[block, head], maxs[block, head]

Matches MLX's sdpa_vector_2pass_1<bfloat16_t, 256, 256> exactly.
"""

import mlx.core as mx


def _gen_sdpa_pass1_source(D=256):
    """Generate Metal source for SDPA Pass 1.

    Replicates MLX's sdpa_vector_2pass_1 for decode (S_q=1, no mask).

    Inputs:
      queries:  [B, H_q, 1, D] bf16  (pre-scaled by 1/sqrt(D) NOT done here)
      keys:     [B, H_kv, alloc_len, D] bf16  (full cache buffer)
      values:   [B, H_kv, alloc_len, D] bf16  (full cache buffer)
      scale:    scalar f32
      KV_alloc_len: int  (allocated cache length, for head stride computation)

    Outputs:
      o_partials: [B, H_q, blocks, D] bf16
      sums:       [B, H_q, blocks] f32
      maxs:       [B, H_q, blocks] f32

    Grid:  (H_kv * 32, gqa_factor, blocks * B)
    TG:    (32, gqa_factor, 1)

    We use grid.z = blocks * B to encode both block_idx and batch_idx:
      block_idx = tgid.z % blocks
      batch_idx = tgid.z / blocks
    """
    EPT = D // 32  # elements per thread = 8

    return f"""
    const int D_DIM = {D};
    const int EPT = {EPT};

    uint simd_lid = thread_index_in_simdgroup;
    uint kv_head_idx = threadgroup_position_in_grid.x;
    uint gqa_lane = thread_index_in_threadgroup / 32;  // which Q head within GQA group

    // Decode block_idx and batch_idx from grid z
    uint block_idx = threadgroup_position_in_grid.z % (uint)N_blocks;
    uint batch_idx = threadgroup_position_in_grid.z / (uint)N_blocks;

    int gqa_factor_val = (int)H_Q / (int)H_KV;
    int q_head_idx = (int)kv_head_idx * gqa_factor_val + (int)gqa_lane;
    int N_val = (int)N_keys;

    // ── Pointer offsets ──
    // queries: [B, H_q, 1, D] — one query per head
    int q_offset = ((int)batch_idx * (int)H_Q + q_head_idx) * D_DIM;

    // keys/values: [B, H_kv, alloc_len, D] — full cache buffer, read only first N_val positions
    // Use alloc_len for head stride (not N_val) to handle non-contiguous cache slices
    int alloc = (int)KV_alloc_len;
    int kv_head_offset = ((int)batch_idx * (int)H_KV + (int)kv_head_idx) * alloc * D_DIM;
    int k_offset = kv_head_offset + (int)block_idx * D_DIM;
    int v_offset = kv_head_offset + (int)block_idx * D_DIM;

    // Output offsets: indexed by [batch, q_head, block]
    int out_head_offset = ((int)batch_idx * (int)H_Q + q_head_idx);
    int o_offset = (out_head_offset * (int)N_blocks + (int)block_idx) * D_DIM;
    int s_offset = out_head_offset * (int)N_blocks + (int)block_idx;

    // ── Load query (once, scaled) ──
    float q[{EPT}];
    for (int i = 0; i < EPT; i++) {{
        q[i] = (float)scale * (float)queries[q_offset + (int)simd_lid * EPT + i];
    }}

    // ── Online softmax + V accumulation ──
    float max_score = -__FLT_MAX__;
    float sum_exp = 0.0f;
    float o[{EPT}] = {{0}};

    int k_stride = (int)N_blocks * D_DIM;  // stride between consecutive keys for this block

    for (int pos = (int)block_idx; pos < N_val; pos += (int)N_blocks) {{
        // Dot product: q @ k[pos]
        float score = 0.0f;
        for (int i = 0; i < EPT; i++) {{
            score += q[i] * (float)keys[k_offset + (int)simd_lid * EPT + i];
        }}
        score = simd_sum(score);

        // Online softmax update
        float new_max = metal::max(max_score, score);
        float factor = metal::fast::exp(max_score - new_max);
        float exp_score = metal::fast::exp(score - new_max);

        max_score = new_max;
        sum_exp = sum_exp * factor + exp_score;

        // Accumulate weighted value
        for (int i = 0; i < EPT; i++) {{
            o[i] = o[i] * factor + exp_score * (float)values[v_offset + (int)simd_lid * EPT + i];
        }}

        // Advance to next position for this block
        k_offset += k_stride;
        v_offset += k_stride;
    }}

    // ── Write partials ──
    if (simd_lid == 0) {{
        sums[s_offset] = sum_exp;
        maxs[s_offset] = max_score;
    }}

    for (int i = 0; i < EPT; i++) {{
        o_partials[o_offset + (int)simd_lid * EPT + i] = static_cast<bfloat16_t>(o[i]);
    }}
"""


_pass1_cache = {}


def _get_pass1_kernel(D, gqa_factor):
    """Get or compile the SDPA Pass 1 kernel."""
    key = (D, gqa_factor)
    if key not in _pass1_cache:
        _pass1_cache[key] = mx.fast.metal_kernel(
            name="custom_sdpa_pass1",
            input_names=["queries", "keys", "values",
                         "scale", "H_Q", "H_KV", "N_keys", "N_blocks",
                         "KV_alloc_len"],
            output_names=["o_partials", "sums", "maxs"],
            source=_gen_sdpa_pass1_source(D),
        )
    return _pass1_cache[key]


def custom_sdpa_pass1(queries, keys, values, scale, H_q, H_kv, D,
                      blocks=128, batch_size=1, N=None, alloc_len=None,
                      scalars=None):
    """SDPA Pass 1: online softmax + partial V accumulation.

    Args:
        queries: [B, H_q, 1, D] bf16
        keys:    [B, H_kv, N_or_alloc, D] bf16 — can be full cache buffer
        values:  [B, H_kv, N_or_alloc, D] bf16 — can be full cache buffer
        scale:   float (1/sqrt(D))
        H_q, H_kv, D: int
        blocks:  int (number of blocks, default 128 for M3 Ultra)
        batch_size: int
        N:         actual sequence length (if None, inferred from keys.shape[2])
        alloc_len: allocated cache length (if None, same as N — contiguous)
        scalars: dict of pre-cached mx.array scalars (optional)

    Returns:
        o_partials: [B, H_q, blocks, D] bf16
        sums:       [B, H_q, blocks] f32
        maxs:       [B, H_q, blocks] f32
    """
    B = batch_size
    gqa_factor = H_q // H_kv

    kern = _get_pass1_kernel(D, gqa_factor)

    if N is None:
        N = keys.shape[2]
    if alloc_len is None:
        alloc_len = N

    if scalars is not None:
        s = scalars
        # N and alloc_len change per call, must create fresh
        n_keys_arr = mx.array(N, dtype=mx.int32)
        alloc_arr = mx.array(alloc_len, dtype=mx.int32)
        inputs = [queries, keys, values,
                  s['scale'], s['H_Q'], s['H_KV'], n_keys_arr, s['N_blocks'],
                  alloc_arr]
    else:
        inputs = [queries, keys, values,
                  mx.array(scale, dtype=mx.float32),
                  mx.array(H_q, dtype=mx.int32),
                  mx.array(H_kv, dtype=mx.int32),
                  mx.array(N, dtype=mx.int32),
                  mx.array(blocks, dtype=mx.int32),
                  mx.array(alloc_len, dtype=mx.int32)]

    results = kern(
        inputs=inputs,
        output_shapes=[
            (B * H_q * blocks * D,),   # o_partials
            (B * H_q * blocks,),       # sums
            (B * H_q * blocks,),       # maxs
        ],
        output_dtypes=[mx.bfloat16, mx.float32, mx.float32],
        grid=(H_kv * 32, gqa_factor, blocks * B),
        threadgroup=(32, gqa_factor, 1),
    )

    o_partials = results[0].reshape(B, H_q, blocks, D)
    sums = results[1].reshape(B, H_q, blocks)
    maxs = results[2].reshape(B, H_q, blocks)
    return o_partials, sums, maxs
