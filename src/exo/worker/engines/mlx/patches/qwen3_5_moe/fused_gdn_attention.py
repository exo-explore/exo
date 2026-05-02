"""Fused GDN attention __call__ for qwen3_5.GatedDeltaNet (Dispatches 2-5).

Replaces the vanilla GatedDeltaNet.__call__ with fused kernel dispatches:
  Dispatch 2: fused_gdn_projections — merged 8-bit GEMV + conv1d + SiLU(qkv) + SiLU(z)
              + sigmoid(b)→beta + g=exp(-exp(A_log)*softplus(a+dt_bias))
  Dispatch 3: fused_qk_rmsnorm — per-head L2-norm on q (×Dk^(-½)) and k
  Dispatch 4: gated_delta_kernel — GDN recurrence (receives pre-computed g, beta)
  Dispatch 5: fused_rms_norm_gated — RMSNorm(out, weight) × z_silu

All 4 projection weights are pre-merged into contiguous buffers at patch time
(_patch_gdn_proj_weights) for better memory locality.

g/beta computation is fused into Dispatch 2 epilogues, eliminating ~8 micro-
dispatches that gated_delta_update would otherwise generate.

Fused path is decode-only (S=1). For prefill (S>1), falls back to vanilla ops.

Returns pre-out_proj output (same interface as _pre_oproj_qwen35_linear_attn_call).
Dispatch 1 (input_layernorm) is handled by the decoder.
Dispatch 6 (oproj_gate_gemv) is handled by the MoE __call__.
"""

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .kernels.batched_fused_gdn_projections_8bit import batched_fused_gdn_projections as fused_gdn_projections
from .kernels.fused_qk_rmsnorm import fused_qk_rmsnorm
from .kernels.fused_rms_norm_gated import fused_rms_norm_gated


def _vanilla_gdn_call(self, inputs, mask, cache):
    """Vanilla GDN path for prefill (S>1). Returns pre-out_proj output."""
    from mlx_lm.models.gated_delta import gated_delta_update

    B, S, _ = inputs.shape

    qkv = self.in_proj_qkv(inputs)
    z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
    b = self.in_proj_b(inputs)
    a = self.in_proj_a(inputs)

    if cache is not None and cache[0] is not None:
        conv_state = cache[0]
    else:
        conv_state = mx.zeros(
            (B, self.conv_kernel_size - 1, self.conv_dim),
            dtype=inputs.dtype,
        )

    if mask is not None:
        qkv = mx.where(mask[..., None], qkv, 0)
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    if cache is not None:
        cache[0] = conv_input[:, -(self.conv_kernel_size - 1):]
    conv_out = nn.silu(self.conv1d(conv_input))

    q, k, v = [
        t.reshape(B, S, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
            [self.num_k_heads, self.num_k_heads, self.num_v_heads],
            [self.head_k_dim, self.head_k_dim, self.head_v_dim],
        )
    ]

    state = cache[1] if cache else None
    inv_scale = k.shape[-1] ** -0.5
    q = inv_scale * q * mx.rsqrt(
        (q * q).sum(axis=-1, keepdims=True) + 1e-6
    )
    k = k * mx.rsqrt(
        (k * k).sum(axis=-1, keepdims=True) + 1e-6
    )

    out, state = gated_delta_update(
        q, k, v, a, b,
        self.A_log, self.dt_bias,
        state, mask,
        use_kernel=True,
    )

    if cache is not None:
        cache[1] = state

    out = self.norm(out, z)
    return self.out_proj(out.reshape(B, S, -1))  # include out_proj for vanilla decoder


def _fused_gdn_call(
    self,
    inputs: mx.array,
    mask: Optional[mx.array] = None,
    cache: Optional[Any] = None,
) -> mx.array:
    """Fused GDN attention: merged projections + existing GDN kernel.

    Decode (S=1): uses fused kernels with merged weight buffers.
    Prefill (S>1): falls back to vanilla ops.

    Returns pre-out_proj output [B, S, value_dim] for Dispatch 6.
    """
    B, S, _ = inputs.shape

    # Vanilla fallback: fused kernels are decode-only (S=1, B<=8)
    if S > 1 or B > 8:
        return _vanilla_gdn_call(self, inputs, mask, cache)

    from mlx_lm.models.gated_delta import gated_delta_kernel

    # ── Cache: conv state ──
    if cache is not None and cache[0] is not None:
        conv_state = cache[0]
    else:
        conv_state = mx.zeros(
            (B, self.conv_kernel_size - 1, self.conv_dim),
            dtype=inputs.dtype,
        )

    # ── Dispatch 2: fused projections (merged GEMV + conv + SiLU + g/beta) ──
    qkv_conv_silu, z_silu, beta, g, conv_state_out = fused_gdn_projections(
        inputs,
        self._merged_proj_w, self._merged_proj_s, self._merged_proj_b,
        self._merged_proj_dims,
        conv_state, self.conv1d.weight,
        self.A_log, self.dt_bias,
        batch_size=B,
    )

    if cache is not None:
        cache[0] = conv_state_out

    # ── Dispatch 3: fused Q/K L2-norm ──
    qk_normed = fused_qk_rmsnorm(qkv_conv_silu, batch_size=B)

    # ── Split q, k from normed output; v from conv output ──
    q = qk_normed[:, :, :self.key_dim].reshape(B, S, self.num_k_heads, self.head_k_dim)
    k = qk_normed[:, :, self.key_dim:].reshape(B, S, self.num_k_heads, self.head_k_dim)
    v = qkv_conv_silu[:, :, 2 * self.key_dim:].reshape(B, S, self.num_v_heads, self.head_v_dim)

    # ── Dispatch 4: GDN recurrence with pre-computed g/beta ──
    state = cache[1] if cache else None
    if state is None:
        state = mx.zeros(
            (B, self.num_v_heads, self.head_v_dim, self.head_k_dim),
            dtype=inputs.dtype,
        )

    out, state_new = gated_delta_kernel(
        q, k, v, g, beta, state, mask,
    )

    if cache is not None:
        cache[1] = state_new

    # ── Dispatch 5: fused RMSNorm × z_silu ──
    norm_weight = self.norm.weight
    result = fused_rms_norm_gated(out, z_silu, norm_weight, batch_size=B)

    return result  # [B, S, value_dim] — skip out_proj (handled by Dispatch 6)


def _fused_gdn_with_outproj_call(self, inputs, mask=None, cache=None):
    """Fused GDN attention path WITH out_proj kept inside attention.

    Wraps _fused_gdn_call (which returns pre-out_proj on the fast path) and
    applies self.out_proj. Used by the fused_attn_batched_moe mode where we
    want fused attention but no oproj cross-boundary fusion — keeps the TP
    all-reduce boundary at out_proj intact.

    The vanilla fallback inside _fused_gdn_call already applies out_proj
    (it serves the legacy batched_fused_oproj mode), so we only apply
    out_proj here on the fast path (S=1, B<=8).
    """
    B, S, _ = inputs.shape
    out = _fused_gdn_call(self, inputs, mask, cache)
    if S > 1 or B > 8:
        return out
    return self.out_proj(out)
