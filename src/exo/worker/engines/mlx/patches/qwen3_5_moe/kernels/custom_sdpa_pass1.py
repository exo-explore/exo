"""Custom SDPA Pass 1 for GQA attention (Dispatch 4).

Online softmax over N key positions, strided by blocks.
Constants baked into Metal source. Per-call varying values (N_keys, alloc_len)
passed via a params array.
"""

import mlx.core as mx


def _gen_sdpa_pass1_source(D=256, H_q=16, H_kv=2, blocks=128):
    EPT = D // 32
    gqa_factor = H_q // H_kv

    return f"""
    const int D_DIM = {D};
    const int EPT = {EPT};
    const int H_Q = {H_q};
    const int H_KV = {H_kv};
    const int N_BLOCKS = {blocks};
    const int GQA_FACTOR = {gqa_factor};

    uint simd_lid = thread_index_in_simdgroup;
    uint kv_head_idx = threadgroup_position_in_grid.x;
    uint gqa_lane = thread_index_in_threadgroup / 32;

    uint block_idx = threadgroup_position_in_grid.z % (uint)N_BLOCKS;
    uint batch_idx = threadgroup_position_in_grid.z / (uint)N_BLOCKS;

    int q_head_idx = (int)kv_head_idx * GQA_FACTOR + (int)gqa_lane;
    int N_val = params[0];
    int alloc = params[1];

    int q_offset = ((int)batch_idx * H_Q + q_head_idx) * D_DIM;

    int kv_head_offset = ((int)batch_idx * H_KV + (int)kv_head_idx) * alloc * D_DIM;
    int k_offset = kv_head_offset + (int)block_idx * D_DIM;
    int v_offset = kv_head_offset + (int)block_idx * D_DIM;

    int out_head_offset = ((int)batch_idx * H_Q + q_head_idx);
    int o_offset = (out_head_offset * N_BLOCKS + (int)block_idx) * D_DIM;
    int s_offset = out_head_offset * N_BLOCKS + (int)block_idx;

    float q[{EPT}];
    for (int i = 0; i < EPT; i++) {{
        q[i] = (float)scale * (float)queries[q_offset + (int)simd_lid * EPT + i];
    }}

    float max_score = -__FLT_MAX__;
    float sum_exp = 0.0f;
    float o[{EPT}] = {{0}};

    int k_stride = N_BLOCKS * D_DIM;

    for (int pos = (int)block_idx; pos < N_val; pos += N_BLOCKS) {{
        float score = 0.0f;
        for (int i = 0; i < EPT; i++) {{
            score += q[i] * (float)keys[k_offset + (int)simd_lid * EPT + i];
        }}
        score = simd_sum(score);

        float new_max = metal::max(max_score, score);
        float factor = metal::fast::exp(max_score - new_max);
        float exp_score = metal::fast::exp(score - new_max);

        max_score = new_max;
        sum_exp = sum_exp * factor + exp_score;

        for (int i = 0; i < EPT; i++) {{
            o[i] = o[i] * factor + exp_score * (float)values[v_offset + (int)simd_lid * EPT + i];
        }}

        k_offset += k_stride;
        v_offset += k_stride;
    }}

    if (simd_lid == 0) {{
        sums[s_offset] = sum_exp;
        maxs[s_offset] = max_score;
    }}

    for (int i = 0; i < EPT; i++) {{
        o_partials[o_offset + (int)simd_lid * EPT + i] = static_cast<bfloat16_t>(o[i]);
    }}
"""


_pass1_cache = {}


def _get_pass1_kernel(D, H_q, H_kv, blocks, gqa_factor):
    key = (D, H_q, H_kv, blocks)
    if key not in _pass1_cache:
        _pass1_cache[key] = mx.fast.metal_kernel(
            name=f"sdpa_pass1_D{D}_Hq{H_q}_Hkv{H_kv}_b{blocks}",
            input_names=["queries", "keys", "values", "scale", "params"],
            output_names=["o_partials", "sums", "maxs"],
            source=_gen_sdpa_pass1_source(D, H_q, H_kv, blocks),
        )
    return _pass1_cache[key]


def custom_sdpa_pass1(queries, keys, values, scale, H_q, H_kv, D,
                      blocks=128, batch_size=1, N=None, alloc_len=None,
                      scalars=None):
    B = batch_size
    gqa_factor = H_q // H_kv

    kern = _get_pass1_kernel(D, H_q, H_kv, blocks, gqa_factor)

    if N is None:
        N = keys.shape[2]
    if alloc_len is None:
        alloc_len = N

    # Pack per-call varying values into a params array (always a pointer)
    params = mx.array([int(N), int(alloc_len)], dtype=mx.int32)

    # Scale as 1-element array (always a pointer)
    scale_arr = mx.array([scale], dtype=mx.float32)

    results = kern(
        inputs=[queries, keys, values, scale_arr, params],
        output_shapes=[
            (B * H_q * blocks * D,),
            (B * H_q * blocks,),
            (B * H_q * blocks,),
        ],
        output_dtypes=[mx.bfloat16, mx.float32, mx.float32],
        grid=(H_kv * 32, gqa_factor, blocks * B),
        threadgroup=(32, gqa_factor, 1),
    )

    o_partials = results[0].reshape(B, H_q, blocks, D)
    sums = results[1].reshape(B, H_q, blocks)
    maxs = results[2].reshape(B, H_q, blocks)
    return o_partials, sums, maxs
