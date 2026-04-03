"""Decoder layer __call__ variants for Qwen3.5.

Two modes:
  _fused_decoder_call: passes residual to fused MoE epilogue (~15 dispatches)
  _oproj_decoder_call: fuses o_proj + RMSNorm + gate GEMV (4 dispatches)

Attention patches for oproj mode:
  _pre_oproj_attention_call: Qwen3NextAttention.__call__ that skips o_proj
  _pre_oproj_qwen35_linear_attn_call: qwen3_5.GatedDeltaNet.__call__ that skips out_proj

Note: qwen3_5.GatedDeltaNet (used by DecoderLayer) is a DIFFERENT class from
qwen3_next.Qwen3NextGatedDeltaNet. They have different projection layouts:
  - qwen3_5.GatedDeltaNet: separate in_proj_qkv, in_proj_z, in_proj_b, in_proj_a
  - qwen3_next.Qwen3NextGatedDeltaNet: merged in_proj_qkvz, in_proj_ba
The patch must match qwen3_5.GatedDeltaNet's __call__ structure.
"""

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.activations import silu as nn_silu

# Map moe block id → parent decoder layer (avoids circular refs in model tree)
_parent_layer_map = {}


def _fused_decoder_call(self, x, mask=None, cache=None):
    """Decoder layer with residual passed to fused MoE epilogue.

    Replaces:
      h = x + attn(norm(x))
      out = h + mlp(norm(h))        # mlp returns MoE output, then adds h

    With:
      h = x + attn(norm(x))
      out = mlp(norm(h), _residual=h)  # epilogue fuses: moe_out + h
    """
    if self.is_linear:
        r = self.linear_attn(self.input_layernorm(x), mask, cache)
    else:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
    h = x + r
    out = self.mlp(self.post_attention_layernorm(h), _residual=h)
    return out  # already includes residual add from epilogue


def _oproj_decoder_call(self, x, mask=None, cache=None):
    """Decoder with fused o_proj + RMSNorm + gate GEMV (oproj 4-dispatch mode).

    Skips o_proj, addmm, and post_attention_layernorm — all fused into Dispatch 1.
    Attention __call__ is patched to return pre-o_proj output.

    Flow:
      pre_oproj = attn(input_layernorm(x))   # returns BEFORE o_proj
      MoE receives (pre_oproj, residual=x) and handles o_proj + RMSNorm + gate internally
    """
    if self.is_linear:
        pre_oproj = self.linear_attn(self.input_layernorm(x), mask, cache)
    else:
        pre_oproj = self.self_attn(self.input_layernorm(x), mask, cache)
    _parent_layer_map[id(self.mlp)] = self
    return self.mlp(pre_oproj, _residual=x)


def _vanilla_decoder_call(self, x, mask=None, cache=None):
    """Original vanilla DecoderLayer.__call__ (fallback for B>8 or S>1)."""
    if self.is_linear:
        r = self.linear_attn(self.input_layernorm(x), mask, cache)
    else:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
    h = x + r
    out = self.mlp(self.post_attention_layernorm(h))
    return h + out


def _fused_gdn_decoder_call(self, x, mask=None, cache=None):
    """Decoder with batched fused kernels. Falls back to vanilla for B>8 or S>1.

    When fused: attention returns pre-out_proj output, MoE handles oproj + gate + experts.
    When vanilla: original DecoderLayer flow (attention + residual + layernorm + MoE).
    """
    B = x.shape[0]
    S = x.shape[1]

    # Full vanilla fallback for large batch or prefill
    if B > 8 or S > 1:
        return _vanilla_decoder_call(self, x, mask, cache)

    # Fused path: attention returns pre-oproj, MoE handles the rest
    if self.is_linear:
        pre_oproj = self.linear_attn(self.input_layernorm(x), mask, cache)
    else:
        pre_oproj = self.self_attn(self.input_layernorm(x), mask, cache)
    _parent_layer_map[id(self.mlp)] = self
    return self.mlp(pre_oproj, _residual=x)


def _pre_oproj_attention_call(self, x, mask=None, cache=None):
    """Qwen3NextAttention.__call__ that returns pre-o_proj output.

    Identical to original except final line returns output*sigmoid(gate)
    instead of self.o_proj(output*sigmoid(gate)).
    """
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


def _pre_oproj_qwen35_linear_attn_call(
    self,
    inputs: mx.array,
    mask: Optional[mx.array] = None,
    cache: Optional[Any] = None,
) -> mx.array:
    """qwen3_5.GatedDeltaNet.__call__ that returns pre-out_proj output.

    Identical to qwen3_5.GatedDeltaNet.__call__ except final line returns
    out.reshape(B,S,-1) instead of self.out_proj(out.reshape(B,S,-1)).

    Note: this targets qwen3_5.GatedDeltaNet (separate projections), NOT
    qwen3_next.Qwen3NextGatedDeltaNet (merged projections). They are
    different classes with different __call__ bodies.
    """
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
        cache[0] = conv_input[:, -(self.conv_kernel_size - 1) :]
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
    q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

    out, state = gated_delta_update(
        q, k, v, a, b,
        self.A_log, self.dt_bias,
        state, mask,
        use_kernel=True,
    )

    if cache is not None:
        cache[1] = state

    out = self.norm(out, z)
    return out.reshape(B, S, -1)  # skip out_proj
