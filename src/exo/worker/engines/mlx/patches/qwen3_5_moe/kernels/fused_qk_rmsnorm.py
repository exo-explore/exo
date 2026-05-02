"""Fused Q/K per-head L2-norm for GDN attention (Dispatch 3).

Performs per-head L2 normalization on q and k vectors with different scaling.
Matches vLLM and latest mlx-lm (qwen3_5.py) which use rsqrt(sum(x²) + eps),
NOT rms_norm which uses rsqrt(mean(x²) + eps).

From qwen3_5.py (updated to match vLLM):
    inv_scale = Dk^(-0.5)
    q = inv_scale * q * rsqrt(sum(q²) + 1e-6)   → L2-normalize then scale by 1/√Dk
    k = k * rsqrt(sum(k²) + 1e-6)                → L2-normalize only (no extra scale)

Grid: (HK_total * 32 threads, 1, B). Each TG = 32 threads = 1 SG, handles one
head (head_k_dim must be 32 * N_READS so a single SG covers it).

All shape constants (key_dim, value_dim, num_k_heads, head_k_dim) are baked
into the Metal source per-shape; we cache one compiled kernel per
(key_dim, value_dim, num_k_heads, head_k_dim) tuple. This lets the same
kernel work at TP=1 (full dims) and TP=N (per-rank sharded dims) without
producing wrong output from stale hardcoded constants.
"""

import mlx.core as mx


def _gen_fused_qk_rmsnorm_source(key_dim, value_dim, num_k_heads, head_k_dim):
    """Generate Metal source for fused Q/K per-head L2-norm.

    Shape conventions (single-mini Qwen3.5-35B-A3B values shown in parens):
      key_dim     = num_k_heads * head_k_dim    (2048 = 16 * 128)
      value_dim   = num_v_heads * head_v_dim    (4096 = 32 * 128)
      n_qkv       = 2*key_dim + value_dim       (8192) — qkv concat width
      n_qk        = 2*key_dim                   (4096) — qk concat width

    Input layout (qkv [B, n_qkv]):
      - [0, key_dim):                  q heads
      - [key_dim, 2*key_dim):          k heads
      - [2*key_dim, n_qkv):            v (untouched, written by Dispatch 4 reader)

    Output layout (qk_out [B, n_qk]):
      - [0, key_dim):                  q L2-normed, scaled by 1/√head_k_dim
      - [key_dim, 2*key_dim):          k L2-normed (no extra scale)

    Grid: (n_heads_total * 32, 1, B), TG (32, 1, 1), where
      n_heads_total = 2 * num_k_heads (q heads then k heads)
      tgid.x in [0, num_k_heads)              → q
      tgid.x in [num_k_heads, 2*num_k_heads)  → k
      tgid.z = batch index
    """
    assert head_k_dim % 32 == 0, f"head_k_dim={head_k_dim} must be divisible by 32"
    n_reads = head_k_dim // 32
    n_qkv = 2 * key_dim + value_dim
    n_qk = 2 * key_dim

    return f"""
    const int N_READS = {n_reads};
    const int DK = {head_k_dim};
    const int HK = {num_k_heads};
    const int KEY_DIM = {key_dim};
    const int N_QKV = {n_qkv};
    const int N_QK = {n_qk};
    const float EPS = 1e-6f;
    const float Q_SCALE = rsqrt(float(DK));      // inv_scale = head_k_dim^(-0.5)
    const float K_SCALE = 1.0f;                  // no extra scale for k

    uint head_idx = threadgroup_position_in_grid.x;
    uint slid = thread_index_in_simdgroup;
    uint b_idx = thread_position_in_grid.z;

    bool is_q = (head_idx < (uint)HK);

    // Input offset: q heads at [0, KEY_DIM), k heads at [KEY_DIM, 2*KEY_DIM)
    int in_base = is_q
        ? (b_idx * N_QKV + head_idx * DK)
        : (b_idx * N_QKV + KEY_DIM + (head_idx - HK) * DK);

    // Output offset: q at [0, KEY_DIM), k at [KEY_DIM, 2*KEY_DIM)
    int out_base = b_idx * N_QK + head_idx * DK;

    // ── Phase 1: Load N_READS elements + sum of squares ──
    float vals[N_READS];
    float partial_sq = 0.0f;
    int elem_base = slid * N_READS;

    for (int i = 0; i < N_READS; i++) {{
        float xi = float(qkv[in_base + elem_base + i]);
        vals[i] = xi;
        partial_sq += xi * xi;
    }}

    // ── Phase 2: simd reduction (32 threads → full sum of head_k_dim elements) ──
    float sum_sq = simd_sum(partial_sq);

    // ── Phase 3: compute L2 inv-norm (NOT rms_norm — no /head_k_dim) ──
    float inv_rms = metal::precise::rsqrt(sum_sq + EPS);

    // ── Phase 4: scale and write ──
    float scale = is_q ? Q_SCALE : K_SCALE;
    float combined = inv_rms * scale;

    for (int i = 0; i < N_READS; i++) {{
        qk_out[out_base + elem_base + i] = static_cast<bfloat16_t>(vals[i] * combined);
    }}
"""


_fused_qk_rmsnorm_cache = {}


def _get_fused_qk_rmsnorm_kernel(key_dim, value_dim, num_k_heads, head_k_dim):
    """Get or compile the fused Q/K RMSNorm kernel for these dims."""
    key = (key_dim, value_dim, num_k_heads, head_k_dim)
    if key not in _fused_qk_rmsnorm_cache:
        _fused_qk_rmsnorm_cache[key] = mx.fast.metal_kernel(
            name=f"fused_qk_rmsnorm_KD{key_dim}_VD{value_dim}_HK{num_k_heads}_DK{head_k_dim}",
            input_names=["qkv"],
            output_names=["qk_out"],
            source=_gen_fused_qk_rmsnorm_source(key_dim, value_dim, num_k_heads, head_k_dim),
        )
    return _fused_qk_rmsnorm_cache[key]


def fused_qk_rmsnorm(qkv_conv_silu, batch_size=1, key_dim=2048, value_dim=4096,
                     num_k_heads=16, head_k_dim=128):
    """Fused Q/K per-head RMSNorm for GDN attention.

    Args:
        qkv_conv_silu: [B, 1, n_qkv] bf16 — post-conv, post-SiLU output from
            Dispatch 2, where n_qkv = 2*key_dim + value_dim.
            Layout: q at [0, key_dim), k at [key_dim, 2*key_dim), v after.
        batch_size: int — batch dimension.
        key_dim: int — num_k_heads * head_k_dim.
        value_dim: int — num_v_heads * head_v_dim (only used to compute n_qkv).
        num_k_heads: int — number of K heads (= number of Q heads).
        head_k_dim: int — per-head K dim. Must be divisible by 32.

    Returns:
        qk_normed: [B, 1, 2*key_dim] bf16 — normed q (first key_dim) and k
            (next key_dim). v is NOT copied; Dispatch 4 reads v directly
            from qkv_conv_silu[:, :, 2*key_dim:].

    Defaults match single-mini Qwen3.5-35B-A3B (key_dim=2048, value_dim=4096,
    num_k_heads=16, head_k_dim=128). Under TP=N, callers must pass per-rank
    values from the GatedDeltaNet instance attributes.
    """
    B = batch_size
    n_qkv = 2 * key_dim + value_dim
    n_qk = 2 * key_dim
    n_heads_total = 2 * num_k_heads  # q heads + k heads (same count)

    kern = _get_fused_qk_rmsnorm_kernel(key_dim, value_dim, num_k_heads, head_k_dim)

    qkv_flat = qkv_conv_silu.reshape(B, n_qkv)
    results = kern(
        inputs=[qkv_flat],
        output_shapes=[(B * n_qk,)],
        output_dtypes=[mx.bfloat16],
        grid=(n_heads_total * 32, 1, B),
        threadgroup=(32, 1, 1),
    )
    return results[0].reshape(B, 1, n_qk)
