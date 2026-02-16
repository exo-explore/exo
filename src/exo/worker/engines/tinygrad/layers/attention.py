import math
from tinygrad import Tensor
from exo.worker.engines.tinygrad.layers.rotary import apply_rope

def linear_forward(x: Tensor, weight: Tensor) -> Tensor:
    return x @ weight.T

def grouped_query_attention(
    x: Tensor,
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    cos_freqs: Tensor,
    sin_freqs: Tensor,
    cache: "KVCache",
    layer_idx: int,
    position_offset: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> Tensor:
    batch, seq_len, _ = x.shape

    q = linear_forward(x, q_proj).reshape(batch, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    k = linear_forward(x, k_proj).reshape(batch, seq_len, num_kv_heads, head_dim).permute(0, 2, 1, 3)
    v = linear_forward(x, v_proj).reshape(batch, seq_len, num_kv_heads, head_dim).permute(0, 2, 1, 3)

    q = apply_rope(q, cos_freqs, sin_freqs, position_offset)
    k = apply_rope(k, cos_freqs, sin_freqs, position_offset)

    k, v = cache.update(layer_idx, k, v)

    if num_kv_heads < num_heads:
        repeat_factor = num_heads // num_kv_heads
        k = k.repeat((1, repeat_factor, 1, 1))
        v = v.repeat((1, repeat_factor, 1, 1))

    scale = 1.0 / math.sqrt(head_dim)
    full_seq = k.shape[2]
    scores = (q @ k.transpose(-2, -1)) * scale

    if seq_len > 1:
        mask = Tensor.ones(seq_len, full_seq).triu(full_seq - seq_len + 1)
        scores = scores + mask * float("-1e9")

    attn_weights = scores.softmax(axis=-1)
    out = attn_weights @ v
    out = out.permute(0, 2, 1, 3).reshape(batch, seq_len, num_heads * head_dim)

    return linear_forward(out, o_proj)
