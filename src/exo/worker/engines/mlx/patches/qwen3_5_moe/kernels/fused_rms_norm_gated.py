"""Fused RMSNormGated for GDN attention (Dispatch 5).

Fuses RMSNorm(out, weight) × z_silu into one kernel.
SiLU on z was already applied in Dispatch 2, so z_silu arrives as f32.

From qwen3_next.py (Qwen3NextRMSNormGated):
    x = rms_norm(hidden_states, weight, eps)     # weight: [head_v_dim]
    gate = silu(z.float())                        # already done in Dispatch 2
    return (gate * x).to(hidden_states.dtype)

Grid: (num_v_heads * 32 threads, 1, B). Each TG = 32 threads = 1 SG, handles
one head (head_v_dim must be 32 * N_READS so a single SG covers it).

Shape constants (num_v_heads, head_v_dim) are baked into the Metal source
per-shape; we cache one compiled kernel per (num_v_heads, head_v_dim) tuple.
"""

import mlx.core as mx


def _gen_fused_rms_norm_gated_source(num_v_heads, head_v_dim):
    """Generate Metal source for fused RMSNormGated.

    Inputs:
      gdn_out: [B, num_v_heads*head_v_dim] bf16  — GDN output, flattened
      z_silu:  [B, num_v_heads*head_v_dim] f32   — post-SiLU z from Dispatch 2
      weight:  [head_v_dim] f32                  — RMSNormGated learned weight

    Output:
      out: [B, num_v_heads*head_v_dim] bf16      — z_silu * rms_norm(gdn_out, weight)

    Grid: (num_v_heads * 32, 1, B), TG (32, 1, 1)
      tgid.x: head index in [0, num_v_heads)
      tgid.z: batch index
    """
    assert head_v_dim % 32 == 0, f"head_v_dim={head_v_dim} must be divisible by 32"
    n_reads = head_v_dim // 32

    return f"""
    const int N_READS = {n_reads};
    const int DV = {head_v_dim};
    const int HV = {num_v_heads};
    const float EPS = 1e-6f;

    uint head_idx = threadgroup_position_in_grid.x;
    uint slid = thread_index_in_simdgroup;
    uint b_idx = thread_position_in_grid.z;

    int base = b_idx * HV * DV + head_idx * DV;
    int elem_base = slid * N_READS;

    // ── Phase 1: Load gdn_out elements + sum of squares ──
    float gdn_vals[N_READS];
    float partial_sq = 0.0f;

    for (int i = 0; i < N_READS; i++) {{
        float xi = float(gdn_out[base + elem_base + i]);
        gdn_vals[i] = xi;
        partial_sq += xi * xi;
    }}

    // ── Phase 2: simd reduction (32 threads → full sum of head_v_dim elements) ──
    float sum_sq = simd_sum(partial_sq);

    // ── Phase 3: compute inv_rms ──
    float inv_rms = metal::precise::rsqrt(sum_sq / float(DV) + EPS);

    // ── Phase 4: RMSNorm × z_silu, write bf16 ──
    for (int i = 0; i < N_READS; i++) {{
        int idx = elem_base + i;
        float w = float(weight[idx]);                           // learned weight[head_v_dim]
        float normed = gdn_vals[i] * inv_rms * w;              // RMSNorm
        float z_val = z_silu[base + idx];                       // already f32, post-SiLU
        out[base + idx] = static_cast<bfloat16_t>(z_val * normed);
    }}
"""


_fused_rms_norm_gated_cache = {}


def _get_fused_rms_norm_gated_kernel(num_v_heads, head_v_dim):
    """Get or compile the fused RMSNormGated kernel for these dims."""
    key = (num_v_heads, head_v_dim)
    if key not in _fused_rms_norm_gated_cache:
        _fused_rms_norm_gated_cache[key] = mx.fast.metal_kernel(
            name=f"fused_rms_norm_gated_HV{num_v_heads}_DV{head_v_dim}",
            input_names=["gdn_out", "z_silu", "weight"],
            output_names=["out"],
            source=_gen_fused_rms_norm_gated_source(num_v_heads, head_v_dim),
        )
    return _fused_rms_norm_gated_cache[key]


def fused_rms_norm_gated(gdn_out, z_silu, weight, batch_size=1,
                          num_v_heads=32, head_v_dim=128):
    """Fused RMSNormGated: RMSNorm(gdn_out, weight) × z_silu.

    Args:
        gdn_out: [B, 1, num_v_heads, head_v_dim] bf16 — GDN recurrence output.
        z_silu:  [B, 1, num_v_heads*head_v_dim] f32 — post-SiLU z from Dispatch 2.
        weight:  [head_v_dim] f32 — RMSNormGated learned weight.
        batch_size: int.
        num_v_heads: int — number of V heads.
        head_v_dim: int — per-head V dim. Must be divisible by 32.

    Returns:
        out: [B, 1, num_v_heads*head_v_dim] bf16 — ready for out_proj.

    Defaults match single-mini Qwen3.5-35B-A3B (num_v_heads=32, head_v_dim=128).
    Under TP=N, callers must pass per-rank values from the GatedDeltaNet
    instance attributes.
    """
    B = batch_size
    value_dim = num_v_heads * head_v_dim

    kern = _get_fused_rms_norm_gated_kernel(num_v_heads, head_v_dim)

    gdn_flat = gdn_out.reshape(B, value_dim)
    z_flat = z_silu.reshape(B, value_dim)

    results = kern(
        inputs=[gdn_flat, z_flat, weight],
        output_shapes=[(B * value_dim,)],
        output_dtypes=[mx.bfloat16],
        grid=(num_v_heads * 32, 1, B),
        threadgroup=(32, 1, 1),
    )
    return results[0].reshape(B, 1, value_dim)
