from __future__ import annotations

import math
from typing import TYPE_CHECKING

from tinygrad.tensor import Tensor

from exo.worker.engines.tinygrad.layers.normalization import rms_norm
from exo.worker.engines.tinygrad.layers.rotary import apply_rope
from exo.worker.engines.tinygrad.quantization.layers import QuantizedLinear

if TYPE_CHECKING:
    from exo.worker.engines.tinygrad.cache import KVCache

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
    cache: "KVCache",
    layer_idx: int,
    position_offset: int,
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

    k, v = cache.update(layer_idx, k, v)

    if num_kv_heads < num_heads:
        repeat_factor = num_heads // num_kv_heads
        k = k.repeat_interleave(repeat_factor, dim = 1)  # pyright: ignore[reportUnknownMemberType]
        v = v.repeat_interleave(repeat_factor, dim = 1)  # pyright: ignore[reportUnknownMemberType]

    scale = 1.0 / math.sqrt(head_dim)
    full_seq: int = int(k.shape[2])
    scores: Tensor = (q @ k.transpose(-2, -1)) * scale

    if seq_len > 1:
        mask = Tensor.ones(seq_len, full_seq).triu(int(full_seq - seq_len + 1))  # pyright: ignore[reportUnknownMemberType]
        scores = scores + mask * float("-1e9")

    attn_weights: Tensor = scores.softmax(axis=-1)
    out: Tensor = attn_weights @ v
    out = out.permute(0, 2, 1, 3).reshape(int(_batch), seq_len, num_heads * head_dim)  # pyright: ignore[reportUnknownMemberType]

    return linear_forward(out, o_proj)
