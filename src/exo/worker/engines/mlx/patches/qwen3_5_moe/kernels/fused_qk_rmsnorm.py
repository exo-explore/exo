"""Fused Q/K per-head L2-norm for GDN attention (Dispatch 3).

Performs per-head L2 normalization on q and k vectors with different scaling.
Matches vLLM and latest mlx-lm (qwen3_5.py) which use rsqrt(sum(x²) + eps),
NOT rms_norm which uses rsqrt(mean(x²) + eps).

From qwen3_5.py (updated to match vLLM):
    inv_scale = Dk^(-0.5) = 128^(-0.5)
    q = inv_scale * q * rsqrt(sum(q²) + 1e-6)   → L2-normalize then scale by 1/√Dk
    k = k * rsqrt(sum(k²) + 1e-6)                → L2-normalize only (no extra scale)

Grid: (32 heads × 32 threads, 1, B).
Each TG = 32 threads = 1 SG, handles one 128-dim head.
Dk=128 = 32 threads × 4 elements → exactly 1 SG, no cross-SG reduction.
"""

import mlx.core as mx

from ..common import COMPUTE_DTYPE, METAL_HALF_TYPE


def _gen_fused_qk_rmsnorm_source():
    """Generate Metal source for fused Q/K per-head L2-norm.

    Input: qkv [B, 8192] bf16 (flattened from [B, 1, 8192])
      - [0, 2048): q = 16 heads × 128
      - [2048, 4096): k = 16 heads × 128
      - [4096, 8192): v (untouched)

    Output: qk_out [B, 4096] bf16
      - [0, 2048): q L2-normalized then scaled by 1/√Dk
      - [2048, 4096): k L2-normalized (no extra scale)

    Grid: (32 * 32, 1, B), TG: (32, 1, 1)
      tgid.x 0..15:  q heads → scale = 1/√128
      tgid.x 16..31: k heads → scale = 1.0
      tgid.z: batch index
    """
    return """
    const int N_READS = 4;
    const int DK = 128;
    const int HK = 16;
    const float EPS = 1e-6f;
    const float Q_SCALE = rsqrt(128.0f);        // inv_scale = Dk^(-0.5)
    const float K_SCALE = 1.0f;                  // no extra scale for k

    uint head_idx = threadgroup_position_in_grid.x;
    uint slid = thread_index_in_simdgroup;
    uint b_idx = thread_position_in_grid.z;

    bool is_q = (head_idx < (uint)HK);

    // Input offset: q heads at [0, 2048), k heads at [2048, 4096)
    int in_base = is_q
        ? (b_idx * 8192 + head_idx * DK)
        : (b_idx * 8192 + 2048 + (head_idx - HK) * DK);

    // Output offset: q at [0, 2048), k at [2048, 4096)
    int out_base = b_idx * 4096 + head_idx * DK;

    // ── Phase 1: Load 4 elements + sum of squares ──
    float vals[4];
    float partial_sq = 0.0f;
    int elem_base = slid * N_READS;

    for (int i = 0; i < N_READS; i++) {
        float xi = float(qkv[in_base + elem_base + i]);
        vals[i] = xi;
        partial_sq += xi * xi;
    }

    // ── Phase 2: simd reduction (32 threads → full sum of 128 elements) ──
    float sum_sq = simd_sum(partial_sq);

    // ── Phase 3: compute L2 inv-norm (NOT rms_norm — no /Dk) ──
    float inv_rms = metal::precise::rsqrt(sum_sq + EPS);

    // ── Phase 4: scale and write ──
    float scale = is_q ? Q_SCALE : K_SCALE;
    float combined = inv_rms * scale;

    for (int i = 0; i < N_READS; i++) {
        qk_out[out_base + elem_base + i] = static_cast<bfloat16_t>(vals[i] * combined);
    }
"""


_fused_qk_rmsnorm_kernel = None


def _get_fused_qk_rmsnorm_kernel():
    """Get or compile the fused Q/K RMSNorm kernel."""
    global _fused_qk_rmsnorm_kernel
    if _fused_qk_rmsnorm_kernel is None:
        _fused_qk_rmsnorm_kernel = mx.fast.metal_kernel(
            name="fused_qk_rmsnorm",
            input_names=["qkv"],
            output_names=["qk_out"],
            source=_gen_fused_qk_rmsnorm_source().replace("bfloat16_t", METAL_HALF_TYPE),
        )
    return _fused_qk_rmsnorm_kernel


def fused_qk_rmsnorm(qkv_conv_silu, batch_size=1):
    """Fused Q/K per-head RMSNorm for GDN attention.

    Args:
        qkv_conv_silu: [B, 1, 8192] bf16 — post-conv, post-SiLU output from Dispatch 2.
            First 2048 = q (16 heads × 128), next 2048 = k, last 4096 = v.
        batch_size: int — batch dimension.

    Returns:
        qk_normed: [B, 1, 4096] bf16 — normalized q (first 2048) and k (next 2048).
            v is NOT copied; Dispatch 4 reads v directly from qkv_conv_silu[:, :, 4096:].
    """
    B = batch_size
    kern = _get_fused_qk_rmsnorm_kernel()

    # Flatten to [B, 8192] for kernel
    qkv_flat = qkv_conv_silu.reshape(B, 8192)

    n_heads = 32  # 16 q + 16 k
    results = kern(
        inputs=[qkv_flat],
        output_shapes=[(B * 4096,)],
        output_dtypes=[COMPUTE_DTYPE],
        grid=(n_heads * 32, 1, B),
        threadgroup=(32, 1, 1),
    )
    return results[0].reshape(B, 1, 4096)
