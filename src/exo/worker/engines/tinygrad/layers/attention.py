from __future__ import annotations

import math

from tinygrad.tensor import Tensor

from exo.worker.engines.tinygrad.cache import KVCache
from exo.worker.engines.tinygrad.layers.normalization import rms_norm
from exo.worker.engines.tinygrad.layers.rotary import apply_rope
from exo.worker.engines.tinygrad.quantization.layers import QuantizedLinear

LinearWeight = Tensor | QuantizedLinear

def linear_forward(x: Tensor, weight: LinearWeight) -> Tensor:
    if isinstance(weight, QuantizedLinear):
        return weight(x)
    return x @ weight.T

def grouped_query_attention(
    x: Tensor,
    q_proj: LinearWeight,
    k_proj: LinearWeight,
    v_proj: LinearWeight,
    o_proj: LinearWeight,
    cos_freqs: Tensor,
    sin_freqs: Tensor,
    cache: KVCache,
    layer_idx: int,
    position_offset: int | Tensor,
    cache_position: int | Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_norm: Tensor | None = None,
    k_norm: Tensor | None = None,
    rms_norm_eps: float = 1e-6,
) -> Tensor:
    _batch, seq_len, _ = x.shape

    q = linear_forward(x, q_proj).reshape(int(_batch), seq_len, num_heads, head_dim).permute(0, 2, 1, 3)  # pyright: ignore[reportUnknownMemberType]
    k = linear_forward(x, k_proj).reshape(int(_batch), seq_len, num_kv_heads, head_dim).permute(0, 2, 1, 3)  # pyright: ignore[reportUnknownMemberType]
    v = linear_forward(x, v_proj).reshape(int(_batch), seq_len, num_kv_heads, head_dim).permute(0, 2, 1, 3)  # pyright: ignore[reportUnknownMemberType]

    if q_norm is not None:
        q = rms_norm(q, q_norm, rms_norm_eps)
    if k_norm is not None:
        k = rms_norm(k, k_norm, rms_norm_eps)

    q = apply_rope(q, cos_freqs, sin_freqs, position_offset)
    k = apply_rope(k, cos_freqs, sin_freqs, position_offset)

    # Store K,V in cache for future decode steps
    cache.update(layer_idx, k, v, position = cache_position)

    if isinstance(position_offset, int):
        # Prefill: compute attention against local K,V (seq_len × seq_len).
        # This avoids the wasteful (seq_len × max_seq_len) matmul that the
        # full-cache path would produce — up to 80× less work for short prompts.
        k_attn, v_attn = k, v
    else:
        # Decode (JIT): use full pre-allocated cache K,V.
        # Shapes are fixed (max_seq_len) which is required for TinyJit replay.
        k_attn = cache._keys[layer_idx]
        v_attn = cache._values[layer_idx]

    if num_kv_heads < num_heads:
        repeat_factor = num_heads // num_kv_heads
        k_attn = k_attn.repeat_interleave(repeat_factor, dim = 1)  # pyright: ignore[reportUnknownMemberType]
        v_attn = v_attn.repeat_interleave(repeat_factor, dim = 1)  # pyright: ignore[reportUnknownMemberType]

    scale = 1.0 / math.sqrt(head_dim)
    scores: Tensor = (q @ k_attn.transpose(-2, -1)) * scale

    if isinstance(position_offset, int):
        # Prefill: standard causal mask over (seq_len × seq_len).
        # All positions in the local K,V are valid, so no unfilled-position mask needed.
        if seq_len > 1:
            causal_mask = Tensor.ones(seq_len, seq_len).triu(1).reshape(1, 1, seq_len, seq_len)  # pyright: ignore[reportUnknownMemberType]
            scores = scores + causal_mask * float("-1e9")
    else:
        # Decode: mask unfilled positions in the full cache.
        valid_len = cache_position + seq_len  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]
        col_indeces: Tensor = cache.col_indices
        unfilled_mask: Tensor = col_indeces >= valid_len  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]
        scores = scores + unfilled_mask * float("-1e9")  # pyright: ignore[reportUnknownVariableType]

    attn_weights: Tensor = scores.softmax(axis=-1)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    out: Tensor = attn_weights @ v_attn  # pyright: ignore[reportUnknownVariableType]
    out = out.permute(0, 2, 1, 3).reshape(int(_batch), seq_len, num_heads * head_dim)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    return linear_forward(out, o_proj)  # pyright: ignore[reportUnknownArgumentType]
