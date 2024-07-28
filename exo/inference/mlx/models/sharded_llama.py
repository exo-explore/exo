from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from exo.inference.shard import Shard
from mlx_lm.models.base import BaseModelArgs, KVCache, create_additive_causal_mask


@dataclass
class NormalModelArgs(BaseModelArgs):
  model_type: str
  hidden_size: int
  num_hidden_layers: int
  intermediate_size: int
  num_attention_heads: int
  rms_norm_eps: float
  vocab_size: int
  head_dim: Optional[int] = None
  max_position_embeddings: Optional[int] = None
  num_key_value_heads: Optional[int] = None
  attention_bias: bool = False
  mlp_bias: bool = False
  rope_theta: float = 10000
  rope_traditional: bool = False
  rope_scaling: Optional[Dict[str, Union[float, str]]] = None
  tie_word_embeddings: bool = True

  def __post_init__(self):
    if self.num_key_value_heads is None:
      self.num_key_value_heads = self.num_attention_heads

    if self.rope_scaling:
      if not "factor" in self.rope_scaling:
        raise ValueError(f"rope_scaling must contain 'factor'")
      rope_type = self.rope_scaling.get("type") or self.rope_scaling.get("rope_type")
      if rope_type is None:
        raise ValueError(f"rope_scaling must contain either 'type' or 'rope_type'")
      if rope_type not in ["linear", "dynamic", "llama3"]:
        raise ValueError("rope_scaling 'type' currently only supports 'linear', 'dynamic' or 'llama3'")


@dataclass
class ModelArgs(NormalModelArgs):
  shard: Shard = field(default_factory=lambda: Shard("", 0, 0, 0))

  def __post_init__(self):
    super().__post_init__()  # Ensure parent initializations are respected

    if isinstance(self.shard, Shard):
      return
    if not isinstance(self.shard, dict):
      raise TypeError(f"Expected shard to be a Shard instance or a dict, got {type(self.shard)} instead")

    self.shard = Shard(**self.shard)


class DynamicNTKScalingRoPE(nn.Module):
  """Implements the rotary positional encoding with Dynamic NTK scaling and Llama 3 RoPE."""

  def __init__(
    self,
    dims: int,
    max_position_embeddings: int = 2048,
    traditional: bool = False,
    base: float = 10000,
    scale: float = 1.0,
    rope_type: str = "default",
    rope_scaling: dict = None,
  ):
    super().__init__()
    self.dims = dims
    self.max_position_embeddings = max_position_embeddings
    self.traditional = traditional
    self.original_base = base
    self.scale = scale
    self.rope_type = rope_type
    self.rope_scaling = rope_scaling
    self.base = self.compute_base_freq()

  def compute_base_freq(self):
    if self.rope_type == "llama3":
      return self.compute_llama3_base_freq()
    return self.original_base

  # source: https://github.com/huggingface/transformers/blob/d5a99dfcee6e94065cb7c83cc8ab6fc5daa0cc4e/src/transformers/modeling_rope_utils.py#L318
  def compute_llama3_base_freq(self):
    factor = self.rope_scaling["factor"]
    low_freq_factor = self.rope_scaling.get("low_freq_factor", 1.0)
    high_freq_factor = self.rope_scaling.get("high_freq_factor", 4.0)
    old_context_len = self.rope_scaling.get(
      "original_max_position_embeddings",
      8192,
    )

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    freqs = self.original_base ** (mx.arange(0, self.dims, 2) / self.dims)
    wavelens = 2 * mx.pi * freqs
    new_base_freqs = []

    smooths = (wavelens - high_freq_wavelen) / (low_freq_wavelen - high_freq_wavelen)
    new_base_freqs = freqs * (1 - smooths) * factor + smooths
    new_base_freqs = mx.where(wavelens < high_freq_wavelen, freqs, new_base_freqs)
    new_base_freqs = mx.where(wavelens > low_freq_wavelen, freqs * factor, new_base_freqs)
    return new_base_freqs.mean().item()

  def extra_repr(self):
    return (
      f"{self.dims}, traditional={self.traditional}, "
      f"max_position_embeddings={self.max_position_embeddings}, "
      f"scaling_factor={self.scale}, rope_type={self.rope_type}"
    )

  def __call__(self, x, offset: int = 0):
    seq_len = x.shape[1] + offset
    base = self.base
    if self.max_position_embeddings and seq_len > self.max_position_embeddings:
      base *= ((self.scale * seq_len / self.max_position_embeddings) - (self.scale - 1)) ** (
        self.dims / (self.dims - 2)
      )

    return mx.fast.rope(
      x,
      self.dims,
      traditional=self.traditional,
      base=base,
      scale=self.scale,
      offset=offset,
    )


def initialize_rope(args: ModelArgs):
  head_dim = args.head_dim or args.hidden_size // args.num_attention_heads

  rope_scaling = args.rope_scaling
  rope_type = "default"
  rope_scale = 1.0

  if rope_scaling is not None:
    rope_type = rope_scaling.get("type") or rope_scaling.get("rope_type") or "default"
    if rope_type == "linear":
      rope_scale = 1 / rope_scaling["factor"]
    elif rope_type == "llama3":
      rope_scale = 1.0  # The scaling is handled internally for llama3

  return DynamicNTKScalingRoPE(
    dims=head_dim,
    max_position_embeddings=args.max_position_embeddings,
    traditional=args.rope_traditional,
    base=args.rope_theta,
    scale=rope_scale,
    rope_type=rope_type,
    rope_scaling=rope_scaling,
  )


class Attention(nn.Module):
  def __init__(self, args: ModelArgs):
    super().__init__()

    dim = args.hidden_size
    self.n_heads = n_heads = args.num_attention_heads
    self.n_kv_heads = n_kv_heads = args.num_key_value_heads

    self.head_dim = head_dim = args.head_dim or args.hidden_size // n_heads

    self.scale = head_dim**-0.5
    if hasattr(args, "attention_bias"):
      attention_bias = args.attention_bias
    else:
      attention_bias = False

    self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=attention_bias)
    self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
    self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
    self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=attention_bias)

    self.rope = initialize_rope(args)

  def __call__(
    self,
    x: mx.array,
    mask: Optional[mx.array] = None,
    cache: Optional[KVCache] = None,
  ) -> mx.array:
    B, L, D = x.shape

    queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

    # Prepare the queries, keys and values for the attention computation
    queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
    keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
    values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

    if cache is not None:
      queries = self.rope(queries, offset=cache.offset)
      keys = self.rope(keys, offset=cache.offset)
      keys, values = cache.update_and_fetch(keys, values)
    else:
      queries = self.rope(queries)
      keys = self.rope(keys)

    output = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=self.scale, mask=mask)
    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return self.o_proj(output)


class MLP(nn.Module):
  def __init__(self, args: ModelArgs):
    super().__init__()

    dim = args.hidden_size
    hidden_dim = args.intermediate_size
    if hasattr(args, "mlp_bias"):
      mlp_bias = args.mlp_bias
    else:
      mlp_bias = False

    self.gate_proj = nn.Linear(dim, hidden_dim, bias=mlp_bias)
    self.down_proj = nn.Linear(hidden_dim, dim, bias=mlp_bias)
    self.up_proj = nn.Linear(dim, hidden_dim, bias=mlp_bias)

  def __call__(self, x) -> mx.array:
    return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
  def __init__(self, args: ModelArgs):
    super().__init__()
    self.num_attention_heads = args.num_attention_heads
    self.hidden_size = args.hidden_size
    self.self_attn = Attention(args)
    self.mlp = MLP(args)
    self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
    self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
    self.args = args

  def __call__(
    self,
    x: mx.array,
    mask: Optional[mx.array] = None,
    cache: Optional[KVCache] = None,
  ) -> mx.array:
    r = self.self_attn(self.input_layernorm(x), mask, cache)
    h = x + r
    r = self.mlp(self.post_attention_layernorm(h))
    out = h + r
    return out


class LlamaModel(nn.Module):
  def __init__(self, args: ModelArgs):
    super().__init__()
    self.args = args
    self.vocab_size = args.vocab_size
    self.num_hidden_layers = args.num_hidden_layers
    assert self.vocab_size > 0
    self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
    self.layers = [TransformerBlock(args=args) for _ in range(args.shard.end_layer - args.shard.start_layer + 1)]
    self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

  def __call__(
    self,
    inputs: mx.array,
    cache=None,
  ):
    if self.args.shard.is_first_layer():
      h = self.embed_tokens(inputs)
    else:
      h = inputs

    mask = None
    if h.shape[1] > 1:
      mask = create_additive_causal_mask(h.shape[1], cache[0].offset if cache is not None else 0)
      mask = mask.astype(h.dtype)

    if cache is None:
      cache = [None] * len(self.layers)

    for layer, c in zip(self.layers, cache):
      h = layer(h, mask, cache=c)

    if self.args.shard.is_last_layer():
      return self.norm(h)
    else:
      return h


class Model(nn.Module):
  def __init__(self, args: ModelArgs):
    super().__init__()
    self.args = args
    self.model_type = args.model_type
    self.model = LlamaModel(args)
    if not args.tie_word_embeddings:
      self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

  def __call__(
    self,
    inputs: mx.array,
    cache=None,
  ):
    out = self.model(inputs, cache)

    if self.args.shard.is_last_layer():
      if self.args.tie_word_embeddings:
        out = self.model.embed_tokens.as_linear(out)
      else:
        out = self.lm_head(out)

    return out

  def sanitize(self, weights):
    # Remove unused precomputed rotary freqs
    return {k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k}

  @property
  def layers(self):
    return self.model.layers

  @property
  def head_dim(self):
    return self.args.head_dim or self.args.hidden_size // self.args.num_attention_heads

  @property
  def n_kv_heads(self):
    return self.args.num_key_value_heads
