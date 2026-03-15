"""Fused Q/K RMSNorm + RoPE for GQA attention (Dispatch 2).

Performs per-head RMSNorm with learned weight (head_dim=256) then applies
RoPE on the first 64 dims (partial_rotary_factor=0.25) using non-traditional
pairing: element p pairs with p+32 (not p+1).

RMSNorm with weight:
    inv_rms = rsqrt(mean(x^2) + eps) = rsqrt(sum(x^2)/D + eps)
    out[i] = x[i] * inv_rms * weight[i]

RoPE (non-traditional, partial):
    Pairs (p, p+32) for p in {0, ..., 31}:
    x'[p]    = x[p]*cos(m*f_p) - x[p+32]*sin(m*f_p)
    x'[p+32] = x[p]*sin(m*f_p) + x[p+32]*cos(m*f_p)
    where f_p = theta^(-p/32), m = cache position
    Only first 64 of 256 dims rotated; remaining 192 unchanged.

Grid: ((H_q + H_kv) * 32, 1, B)
Each TG = 32 threads = 1 SG, handles one 256-dim head.
D=256 = 32 threads x 8 elements -> exactly 1 SG, no cross-SG reduction.
"""

import mlx.core as mx


def _gen_fused_qk_norm_rope_source(H_q=16, H_kv=2, D=256, rope_dims=64):
    """Generate Metal source for fused Q/K RMSNorm + RoPE.

    Input: queries [B, H_q*D] bf16, keys [B, H_kv*D] bf16
    Output: q_out [B, H_q*D] bf16, k_out [B, H_kv*D] bf16

    TG assignment:
      tgid.x 0..H_q-1:              Q heads
      tgid.x H_q..H_q+H_kv-1:       K heads

    Thread layout (32 threads, N_READS=8):
      Thread t handles elements [8t, 8t+7]

    RoPE thread assignment (non-traditional pairing):
      Threads 0-3: first half of pair (elements 0-31)
      Threads 4-7: second half of pair (elements 32-63)
      Thread t paired with t^4 via simd_shuffle
      Threads 8-31: elements 64-255 (unrotated, skip RoPE)
    """
    N_READS = D // 32
    ROPE_HALF = rope_dims // 2          # 32 = number of rotation pairs
    ROPE_THREADS = rope_dims // N_READS  # 8 = threads touching rotated dims
    FIRST_HALF = ROPE_THREADS // 2       # 4 = threads in first half of pairs
    PARTNER_XOR = FIRST_HALF             # 4 = XOR to find partner thread

    return f"""
    // ── Constants ──
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

    // ── Phase 2: RMSNorm reduction (32 threads -> full sum of {D} elements) ──
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

    // ── Phase 4: RoPE on first {rope_dims} dims (threads 0..{ROPE_THREADS - 1}) ──
    if (slid < {ROPE_THREADS}u) {{
        ushort partner = (ushort)(slid ^ {PARTNER_XOR}u);
        int cos_base = (int)(slid & {FIRST_HALF - 1}u) * N_READS;

        // Compute cos/sin on-device from precomputed inv_freq
        // angle = position * inv_freq[d], where inv_freq[d] = theta^(-d/{ROPE_HALF})
        float cos_arr[{N_READS}], sin_arr[{N_READS}];
        for (int i = 0; i < N_READS; i++) {{
            float pos_f = position;
            float angle = pos_f * inv_freq[cos_base + i];
            cos_arr[i] = metal::fast::cos(angle);
            sin_arr[i] = metal::fast::sin(angle);
        }}

        // Exchange normalized values with partner thread via simd_shuffle
        float partner_vals[{N_READS}];
        for (int i = 0; i < N_READS; i++)
            partner_vals[i] = simd_shuffle(vals[i], partner);

        if (slid < {FIRST_HALF}u) {{
            // First half of pair: x'[p] = x[p]*cos - x[p+{ROPE_HALF}]*sin
            for (int i = 0; i < N_READS; i++)
                vals[i] = vals[i] * cos_arr[i] - partner_vals[i] * sin_arr[i];
        }} else {{
            // Second half: x'[p+{ROPE_HALF}] = x_first*sin + x[p+{ROPE_HALF}]*cos
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
    """Get or compile the fused Q/K RMSNorm + RoPE kernel."""
    key = (H_q, H_kv, D, rope_dims)
    if key not in _kernel_cache:
        _kernel_cache[key] = mx.fast.metal_kernel(
            name="fused_qk_norm_rope",
            input_names=["queries", "keys", "q_norm_w", "k_norm_w",
                         "inv_freq", "position"],
            output_names=["q_out", "k_out"],
            source=_gen_fused_qk_norm_rope_source(H_q, H_kv, D, rope_dims),
        )
    return _kernel_cache[key]


def fused_qk_norm_rope(queries, keys, q_norm_weight, k_norm_weight,
                        inv_freq, cache_offset, H_q, H_kv, D,
                        batch_size=1):
    """Fused Q/K per-head RMSNorm + RoPE for GQA attention.

    Args:
        queries: [B, 1, H_q*D] bf16 — raw queries from Dispatch 1.
        keys: [B, 1, H_kv*D] bf16 — raw keys from Dispatch 1.
        q_norm_weight: [D] bf16 — RMSNorm learned weight for queries.
        k_norm_weight: [D] bf16 — RMSNorm learned weight for keys.
        inv_freq: [rope_dims/2] f32 — precomputed theta^(-d/half_dims).
        cache_offset: int — sequence position for RoPE angles.
        H_q: int — number of query heads.
        H_kv: int — number of key/value heads.
        D: int — head dimension.
        batch_size: int — batch size.

    Returns:
        q_normed_roped: [B, H_q, 1, D] bf16 — ready for SDPA.
        k_normed_roped: [B, H_kv, 1, D] bf16 — ready for cache update.
    """
    B = batch_size
    rope_dims = inv_freq.shape[0] * 2

    kern = _get_kernel(H_q, H_kv, D, rope_dims)

    q_flat = queries.reshape(B, H_q * D)
    k_flat = keys.reshape(B, H_kv * D)
    pos = mx.array(cache_offset, dtype=mx.int32)

    n_heads = H_q + H_kv
    results = kern(
        inputs=[q_flat, k_flat, q_norm_weight, k_norm_weight, inv_freq, pos],
        output_shapes=[(B * H_q * D,), (B * H_kv * D,)],
        output_dtypes=[mx.bfloat16, mx.bfloat16],
        grid=(n_heads * 32, 1, B),
        threadgroup=(32, 1, 1),
    )

    q_out = results[0].reshape(B, H_q, 1, D)
    k_out = results[1].reshape(B, H_kv, 1, D)
    return q_out, k_out
