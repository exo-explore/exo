"""Fused Q/K RMSNorm + RoPE for GQA attention (Dispatch 2).

Performs per-head RMSNorm with learned weight (head_dim=256) then applies
RoPE on the first 64 dims (partial_rotary_factor=0.25) using non-traditional
pairing: element p pairs with p+32 (not p+1).

cos/sin values precomputed in Python from inv_freq * position,
passed as array inputs (no scalar kernel inputs).

All constants baked into Metal source (no scalar kernel inputs).
"""

import mlx.core as mx


def _gen_fused_qk_norm_rope_source(H_q=16, H_kv=2, D=256, rope_dims=64):
    N_READS = D // 32
    ROPE_HALF = rope_dims // 2
    ROPE_THREADS = rope_dims // N_READS
    FIRST_HALF = ROPE_THREADS // 2
    PARTNER_XOR = FIRST_HALF

    return f"""
    const int H_Q = {H_q};
    const int H_KV = {H_kv};
    const int D_DIM = {D};
    const int N_READS = {N_READS};
    const float EPS = 1e-6f;

    uint head_idx = threadgroup_position_in_grid.x;
    uint slid = thread_index_in_simdgroup;
    uint b_idx = thread_position_in_grid.z;

    bool is_q = (head_idx < (uint)H_Q);
    int head_local = is_q ? (int)head_idx : ((int)head_idx - H_Q);

    // ── Phase 1: Load N_READS elements + partial sum of squares ──
    int in_base = is_q
        ? (int)(b_idx * H_Q * D_DIM + (int)head_idx * D_DIM)
        : (int)(b_idx * H_KV * D_DIM + head_local * D_DIM);

    int elem_base = (int)slid * N_READS;
    float vals[{N_READS}];
    float partial_sq = 0.0f;

    for (int i = 0; i < N_READS; i++) {{
        float xi;
        if (is_q)
            xi = (float)queries[in_base + elem_base + i];
        else
            xi = (float)keys[in_base + elem_base + i];
        vals[i] = xi;
        partial_sq += xi * xi;
    }}

    // ── Phase 2: RMSNorm reduction ──
    float sum_sq = simd_sum(partial_sq);
    float inv_rms = metal::precise::rsqrt(sum_sq / (float)D_DIM + EPS);

    // ── Phase 3: Normalize with learned weight ──
    for (int i = 0; i < N_READS; i++) {{
        float w;
        if (is_q)
            w = (float)q_norm_w[elem_base + i];
        else
            w = (float)k_norm_w[elem_base + i];
        vals[i] = vals[i] * inv_rms * w;
    }}

    // ── Phase 4: RoPE on first {rope_dims} dims ──
    if (slid < {ROPE_THREADS}u) {{
        ushort partner = (ushort)(slid ^ {PARTNER_XOR}u);
        int cos_base = (int)(slid & {FIRST_HALF - 1}u) * N_READS;

        // Load precomputed cos/sin (computed in Python from position * inv_freq)
        float cos_arr[{N_READS}], sin_arr[{N_READS}];
        for (int i = 0; i < N_READS; i++) {{
            cos_arr[i] = rope_cos[cos_base + i];
            sin_arr[i] = rope_sin[cos_base + i];
        }}

        float partner_vals[{N_READS}];
        for (int i = 0; i < N_READS; i++)
            partner_vals[i] = simd_shuffle(vals[i], partner);

        if (slid < {FIRST_HALF}u) {{
            for (int i = 0; i < N_READS; i++)
                vals[i] = vals[i] * cos_arr[i] - partner_vals[i] * sin_arr[i];
        }} else {{
            for (int i = 0; i < N_READS; i++)
                vals[i] = partner_vals[i] * sin_arr[i] + vals[i] * cos_arr[i];
        }}
    }}

    // ── Phase 5: Write output ──
    int out_base = is_q
        ? (int)(b_idx * H_Q * D_DIM + (int)head_idx * D_DIM)
        : (int)(b_idx * H_KV * D_DIM + head_local * D_DIM);

    for (int i = 0; i < N_READS; i++) {{
        if (is_q)
            q_out[out_base + elem_base + i] = static_cast<bfloat16_t>(vals[i]);
        else
            k_out[out_base + elem_base + i] = static_cast<bfloat16_t>(vals[i]);
    }}
"""


_kernel_cache = {}


def _get_kernel(H_q, H_kv, D, rope_dims):
    key = (H_q, H_kv, D, rope_dims)
    if key not in _kernel_cache:
        _kernel_cache[key] = mx.fast.metal_kernel(
            name=f"fused_qk_norm_rope_Hq{H_q}_Hkv{H_kv}_D{D}_rd{rope_dims}",
            input_names=["queries", "keys", "q_norm_w", "k_norm_w",
                         "rope_cos", "rope_sin"],
            output_names=["q_out", "k_out"],
            source=_gen_fused_qk_norm_rope_source(H_q, H_kv, D, rope_dims),
        )
    return _kernel_cache[key]


def fused_qk_norm_rope(queries, keys, q_norm_weight, k_norm_weight,
                        inv_freq, cache_offset, H_q, H_kv, D,
                        batch_size=1):
    B = batch_size
    rope_dims = inv_freq.shape[0] * 2

    kern = _get_kernel(H_q, H_kv, D, rope_dims)

    q_flat = queries.reshape(B, H_q * D)
    k_flat = keys.reshape(B, H_kv * D)

    # Precompute cos/sin in Python (avoids scalar kernel input)
    angles = float(cache_offset) * inv_freq
    rope_cos = mx.cos(angles).astype(mx.float32)
    rope_sin = mx.sin(angles).astype(mx.float32)

    n_heads = H_q + H_kv
    results = kern(
        inputs=[q_flat, k_flat, q_norm_weight, k_norm_weight, rope_cos, rope_sin],
        output_shapes=[(B * H_q * D,), (B * H_kv * D,)],
        output_dtypes=[mx.bfloat16, mx.bfloat16],
        grid=(n_heads * 32, 1, B),
        threadgroup=(32, 1, 1),
    )

    q_out = results[0].reshape(B, H_q, 1, D)
    k_out = results[1].reshape(B, H_kv, 1, D)
    return q_out, k_out
