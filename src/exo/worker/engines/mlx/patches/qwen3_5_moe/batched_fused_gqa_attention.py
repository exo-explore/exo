"""Batched fused GQA attention for Qwen3.5 (projections + norm+rope fused, vanilla SDPA).

Dispatches:
  1. batched_fused_gqa_projections — merged q+gate+k+v GEMV with register weight sharing
  2. fused_qk_norm_rope — per-head RMSNorm + RoPE (already supports B>1 via grid z)
  3. Vanilla cache update (BatchKVCache)
  4. Vanilla SDPA (MLX built-in, handles batching natively)
  5. Vanilla gate multiply

Returns pre-out_proj output for the oproj MoE block.
Falls back to vanilla (with o_proj) for B>8 or S>1.
"""

from typing import Any, Optional

import mlx.core as mx

from .kernels.batched_fused_gqa_projections_8bit import batched_fused_gqa_projections


def _batched_fused_gqa_call(
    self,
    x: mx.array,
    mask: Optional[mx.array] = None,
    cache: Optional[Any] = None,
) -> mx.array:
    """Batched fused GQA attention with custom projection + norm/rope kernels.

    For 1<=B<=8, S=1: fused projections + fused norm+rope + vanilla SDPA.
    For B>8 or S>1: vanilla fallback.
    Returns pre-out_proj output [B, S, H_q*D].
    """
    B, S, _ = x.shape

    if S > 1 or B > 8:
        # Vanilla fallback
        q_proj_output = self.q_proj(x)
        queries, gate = mx.split(
            q_proj_output.reshape(B, S, self.num_attention_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, S, -1)
        keys, values = self.k_proj(x), self.v_proj(x)
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(
            keys.reshape(B, S, self.num_key_value_heads, -1)
        ).transpose(0, 2, 1, 3)
        values = values.reshape(B, S, self.num_key_value_heads, -1).transpose(
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
        output = output.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.o_proj(output * mx.sigmoid(gate))

    H_q = self.num_attention_heads
    H_kv = self.num_key_value_heads
    D = self.head_dim

    # ── Dispatch 1: batched fused projections ──
    queries, gate_sigmoid, keys, values = batched_fused_gqa_projections(
        x,
        self._merged_proj_w, self._merged_proj_s, self._merged_proj_b,
        self._merged_proj_dims,
        batch_size=B,
        total_tg=getattr(self, '_d1_total_tg', None),
    )

    # ── Dispatch 2+: vanilla norm + rope (avoids mx.eval sync on BatchKVCache offset) ──
    queries = self.q_norm(queries.reshape(B, 1, H_q, D)).transpose(0, 2, 1, 3)
    keys = self.k_norm(keys.reshape(B, 1, H_kv, D)).transpose(0, 2, 1, 3)
    values = values.reshape(B, 1, H_kv, D).transpose(0, 2, 1, 3)

    if cache is not None:
        queries = self.rope(queries, offset=cache.offset)
        keys = self.rope(keys, offset=cache.offset)
    else:
        queries = self.rope(queries)
        keys = self.rope(keys)

    # ── Dispatch 3: KV cache update ──
    if cache is not None:
        keys, values = cache.update_and_fetch(keys, values)

    # ── Dispatch 4: vanilla SDPA ──
    from mlx_lm.models.qwen3_next import scaled_dot_product_attention
    output = scaled_dot_product_attention(
        queries, keys, values, cache=cache, scale=self.scale, mask=mask
    )
    output = output.transpose(0, 2, 1, 3).reshape(B, S, -1)

    # ── Gate multiply ──
    return output * gate_sigmoid.astype(output.dtype)
