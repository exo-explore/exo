"""Fused GQA attention __call__ for qwen3_next.Qwen3NextAttention.

Replaces vanilla Qwen3NextAttention.__call__ with fused kernel dispatches:
  Dispatch 1: fused_gqa_projections — merged 8-bit GEMV (q+gate+k+v) + sigmoid(gate)
  Dispatch 2: fused_qk_norm_rope — RMSNorm + RoPE (TODO: custom kernel)
  Dispatch 3: KV cache update (MLX built-in)
  Dispatch 4: custom_sdpa_pass1 (TODO: custom kernel)
  Dispatch 5: custom_sdpa_pass2_gate (TODO: custom kernel, includes gate multiply)
  Dispatch 6: oproj_gate_gemv (existing, handled by MoE __call__)

Dispatches 1-2, 4-5 are custom kernels, Dispatch 3 is MLX built-in.
Returns pre-out_proj output (output * sigmoid(gate)) for Dispatch 6.

Fused path is decode-only (S=1). For prefill (S>1), falls back to vanilla ops.
"""

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .kernels.fused_gqa_projections_8bit import fused_gqa_projections
from .kernels.fused_qk_norm_rope import fused_qk_norm_rope
from .kernels.custom_sdpa_pass1 import custom_sdpa_pass1
from .kernels.custom_sdpa_pass2_gate import custom_sdpa_pass2_gate


def _vanilla_gqa_call(self, x, mask, cache):
    """Vanilla GQA path for prefill (S>1). Returns pre-out_proj output."""
    B, L, D = x.shape
    q_proj_output = self.q_proj(x)
    queries, gate = mx.split(
        q_proj_output.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1
    )
    gate = gate.reshape(B, L, -1)
    keys, values = self.k_proj(x), self.v_proj(x)
    queries = self.q_norm(queries).transpose(0, 2, 1, 3)
    keys = self.k_norm(
        keys.reshape(B, L, self.num_key_value_heads, -1)
    ).transpose(0, 2, 1, 3)
    values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
        0, 2, 1, 3
    )
    if cache is not None:
        queries = self.rope(queries, offset=cache.offset)
        keys = self.rope(keys, offset=cache.offset)
        keys, values = cache.update_and_fetch(keys, values)
    else:
        queries = self.rope(queries)
        keys = self.rope(keys)

    from mlx_lm.models.qwen3_next import scaled_dot_product_attention
    output = scaled_dot_product_attention(
        queries, keys, values, cache=cache, scale=self.scale, mask=mask
    )
    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return output * mx.sigmoid(gate)  # skip o_proj


def _fused_gqa_call(
    self,
    x: mx.array,
    mask: Optional[mx.array] = None,
    cache: Optional[Any] = None,
) -> mx.array:
    """Fused GQA attention: custom projection + norm/rope kernels.

    Decode (S=1): Dispatch 1 (fused GEMV) + Dispatch 2 (fused norm+rope)
                  + vanilla cache + SDPA (Dispatches 3-5).
    Prefill (S>1): falls back to fully vanilla ops.

    Returns pre-out_proj output [B, S, H_q*D] for Dispatch 6.
    """
    B, S, _ = x.shape

    # Prefill fallback: fused kernels are decode-only (S=1)
    if S > 1:
        return _vanilla_gqa_call(self, x, mask, cache)

    H_q = self.num_attention_heads
    H_kv = self.num_key_value_heads
    D = self.head_dim

    # ── Dispatch 1: fused projections (merged GEMV + sigmoid(gate)) ──
    sc = getattr(self, '_kernel_scalars', None)
    queries, gate_sigmoid, keys, values = fused_gqa_projections(
        x,
        self._merged_proj_w, self._merged_proj_s, self._merged_proj_b,
        self._merged_proj_dims,
        batch_size=B,
        scalars=sc, total_tg=getattr(self, '_d1_total_tg', None),
    )

    # ── Dispatch 2: fused Q/K RMSNorm + RoPE ──
    queries, keys = fused_qk_norm_rope(
        queries, keys,
        self.q_norm.weight, self.k_norm.weight,
        self._rope_inv_freq, cache.offset,
        H_q, H_kv, D, batch_size=B,
    )
    # queries: [B, H_q, 1, D], keys: [B, H_kv, 1, D]
    # Reshape directly to (B, H_kv, 1, D) — no transpose needed since S=1.
    # Avoids a 4 MiB copy dispatch that transpose would trigger.
    values = values.reshape(B, H_kv, 1, D)

    # ── Dispatch 3: KV cache update ──
    cache.update_and_fetch(keys, values)
    N = cache.offset  # actual sequence length after update
    alloc_len = cache.keys.shape[2]  # allocated buffer length

    # ── Dispatch 4: SDPA Pass 1 (online softmax + partial V accumulation) ──
    blocks = 128  # M3 Ultra default for N >= 1024
    if N < 1024:
        # Short sequence: fall back to vanilla SDPA + gate multiply
        # Use sliced views for built-in SDPA (handles strides natively)
        k_sliced = cache.keys[:, :, :N, :]
        v_sliced = cache.values[:, :, :N, :]
        from mlx_lm.models.qwen3_next import scaled_dot_product_attention
        output = scaled_dot_product_attention(
            queries, k_sliced, v_sliced, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return output * gate_sigmoid.astype(output.dtype)

    # Pass full (contiguous) cache buffers + alloc_len to avoid copy dispatch
    o_partials, sums, maxs = custom_sdpa_pass1(
        queries, cache.keys, cache.values, self.scale,
        H_q, H_kv, D, blocks=blocks, batch_size=B,
        N=N, alloc_len=alloc_len, scalars=sc,
    )

    # ── Dispatch 5: SDPA Pass 2 + gate multiply ──
    return custom_sdpa_pass2_gate(
        o_partials, sums, maxs, gate_sigmoid,
        H_q, D, blocks=blocks, V_SPLIT=4, batch_size=B,
        scalars=sc,
    )
