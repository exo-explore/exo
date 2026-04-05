"""Fused RMSNormGated for GDN attention (Dispatch 5).

Fuses RMSNorm(out, weight) × z_silu into one kernel.
SiLU on z was already applied in Dispatch 2, so z_silu arrives as f32.

From qwen3_next.py (Qwen3NextRMSNormGated):
    x = rms_norm(hidden_states, weight, eps)     # weight: [Dv=128]
    gate = silu(z.float())                        # already done in Dispatch 2
    return (gate * x).to(hidden_states.dtype)

Grid: (32 heads × 32 threads, 1, B).
Each TG = 32 threads = 1 SG, handles one 128-dim head.
Dv=128 = 32 threads × 4 elements → exactly 1 SG.
"""

import mlx.core as mx

from ..common import COMPUTE_DTYPE, METAL_HALF_TYPE


def _gen_fused_rms_norm_gated_source():
    """Generate Metal source for fused RMSNormGated.

    Inputs:
      gdn_out: [B, Hv*Dv] bf16  — GDN output, flattened (Hv=32, Dv=128)
      z_silu:  [B, Hv*Dv] f32   — post-SiLU z from Dispatch 2
      weight:  [Dv] f32         — RMSNormGated learned weight (128 elements)

    Output:
      out: [B, Hv*Dv] bf16      — result = z_silu * rms_norm(gdn_out, weight)

    Grid: (32 * 32, 1, B), TG: (32, 1, 1)
      tgid.x: head index (0..31)
      tgid.z: batch index
    """
    return """
    const int N_READS = 4;
    const int DV = 128;
    const int HV = 32;
    const float EPS = 1e-6f;

    uint head_idx = threadgroup_position_in_grid.x;
    uint slid = thread_index_in_simdgroup;
    uint b_idx = thread_position_in_grid.z;

    int base = b_idx * HV * DV + head_idx * DV;
    int elem_base = slid * N_READS;

    // ── Phase 1: Load gdn_out elements + sum of squares ──
    float gdn_vals[4];
    float partial_sq = 0.0f;

    for (int i = 0; i < N_READS; i++) {
        float xi = float(gdn_out[base + elem_base + i]);
        gdn_vals[i] = xi;
        partial_sq += xi * xi;
    }

    // ── Phase 2: simd reduction (32 threads → full sum of 128 elements) ──
    float sum_sq = simd_sum(partial_sq);

    // ── Phase 3: compute inv_rms ──
    float inv_rms = metal::precise::rsqrt(sum_sq / float(DV) + EPS);

    // ── Phase 4: RMSNorm × z_silu, write bf16 ──
    for (int i = 0; i < N_READS; i++) {
        int idx = elem_base + i;
        float w = float(weight[idx]);                           // learned weight[Dv]
        float normed = gdn_vals[i] * inv_rms * w;              // RMSNorm
        float z_val = z_silu[base + idx];                       // already f32, post-SiLU
        out[base + idx] = static_cast<bfloat16_t>(z_val * normed);
    }
"""


_fused_rms_norm_gated_kernel = None


def _get_fused_rms_norm_gated_kernel():
    """Get or compile the fused RMSNormGated kernel."""
    global _fused_rms_norm_gated_kernel
    if _fused_rms_norm_gated_kernel is None:
        _fused_rms_norm_gated_kernel = mx.fast.metal_kernel(
            name="fused_rms_norm_gated",
            input_names=["gdn_out", "z_silu", "weight"],
            output_names=["out"],
            source=_gen_fused_rms_norm_gated_source().replace("bfloat16_t", METAL_HALF_TYPE),
        )
    return _fused_rms_norm_gated_kernel


def fused_rms_norm_gated(gdn_out, z_silu, weight, batch_size=1):
    """Fused RMSNormGated: RMSNorm(out, weight) × z_silu.

    Args:
        gdn_out: [B, 1, Hv, Dv] bf16 — GDN recurrence output (Hv=32, Dv=128).
        z_silu:  [B, 1, 4096] f32 — post-SiLU z from Dispatch 2.
        weight:  [128] f32 — RMSNormGated learned weight (Dv elements).
        batch_size: int.

    Returns:
        out: [B, 1, 4096] bf16 — ready for out_proj in Dispatch 6.
    """
    B = batch_size
    kern = _get_fused_rms_norm_gated_kernel()

    # Flatten to [B, 4096]
    gdn_flat = gdn_out.reshape(B, 4096)
    z_flat = z_silu.reshape(B, 4096)

    n_heads = 32  # Hv
    results = kern(
        inputs=[gdn_flat, z_flat, weight],
        output_shapes=[(B * 4096,)],
        output_dtypes=[COMPUTE_DTYPE],
        grid=(n_heads * 32, 1, B),
        threadgroup=(32, 1, 1),
    )
    return results[0].reshape(B, 1, 4096)
