# Copyright © 2024 Apple Inc.

import math
import inspect
from dataclasses import dataclass, field
from typing import Optional, Dict, Union

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import BaseModelArgs, KVCache
from exo.inference.shard import Shard
from .base import IdentityBlock
import numpy as np


@dataclass
class VisionConfig:
  model_type: str
  num_hidden_layers: int = 24
  hidden_size: int = 1024
  intermediate_size: int = 4096
  num_attention_heads: int = 16
  image_size: int = 336
  patch_size: int = 14
  projection_dim: int = 768
  vocab_size: int = 32000
  num_channels: int = 3
  layer_norm_eps: float = 1e-5

  @classmethod
  def from_dict(cls, params):
    return cls(**{k: v for k, v in params.items() if k in inspect.signature(cls).parameters})


class VisionAttention(nn.Module):
  def __init__(
    self,
    dims: int,
    num_heads: int,
    query_input_dims: Optional[int] = None,
    key_input_dims: Optional[int] = None,
    value_input_dims: Optional[int] = None,
    value_dims: Optional[int] = None,
    value_output_dims: Optional[int] = None,
    bias: bool = False,
  ):
    super().__init__()

    if (dims % num_heads) != 0:
      raise ValueError("The input feature dimensions should be divisible by the "
                       f"number of heads ({dims} % {num_heads}) != 0")

    query_input_dims = query_input_dims or dims
    key_input_dims = key_input_dims or dims
    value_input_dims = value_input_dims or key_input_dims
    value_dims = value_dims or dims
    value_output_dims = value_output_dims or dims

    self.num_heads = num_heads
    self.q_proj = nn.Linear(query_input_dims, dims, bias=bias)
    self.k_proj = nn.Linear(key_input_dims, dims, bias=bias)
    self.v_proj = nn.Linear(value_input_dims, value_dims, bias=bias)
    self.out_proj = nn.Linear(value_dims, value_output_dims, bias=bias)

  def __call__(self, queries, keys, values, mask=None):
    queries = self.q_proj(queries)
    keys = self.k_proj(keys)
    values = self.v_proj(values)

    num_heads = self.num_heads
    B, L, D = queries.shape
    _, S, _ = keys.shape
    queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
    keys = keys.reshape(B, S, num_heads, -1).transpose(0, 2, 3, 1)
    values = values.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)

    scale = math.sqrt(1/queries.shape[-1])
    scores = (queries*scale) @ keys
    if mask is not None:
      scores = scores + mask.astype(scores.dtype)
    scores = mx.softmax(scores, axis=-1)
    values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)

    return self.out_proj(values_hat)


class VisionMLP(nn.Module):
  def __init__(self, config: VisionConfig):
    super().__init__()
    self.activation_fn = nn.GELU(approx="fast")
    self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
    self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

  def __call__(self, x: mx.array) -> mx.array:
    x = self.activation_fn(self.fc1(x))
    x = self.fc2(x)
    return x


class VisionEncoderLayer(nn.Module):
  def __init__(self, config: VisionConfig):
    super().__init__()
    self.embed_dim = config.hidden_size
    self.self_attn = VisionAttention(config.hidden_size, config.num_attention_heads, bias=True)
    self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
    self.mlp = VisionMLP(config)
    self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

  def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
    y = self.layer_norm1(x)
    y = self.self_attn(y, y, y, mask)
    x = x + y
    y = self.layer_norm2(x)
    y = self.mlp(y)
    return x + y


class VisionEncoder(nn.Module):
  def __init__(self, config: VisionConfig):
    super().__init__()
    self.layers = [VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)]


class VisionEmbeddings(nn.Module):
  def __init__(self, config: VisionConfig):
    super().__init__()
    self.config = config
    self.embed_dim = config.hidden_size
    self.image_size = config.image_size
    self.patch_size = config.patch_size

    self.class_embedding = mx.zeros((config.hidden_size,))

    self.patch_embedding = nn.Conv2d(
      in_channels=config.num_channels,
      out_channels=self.embed_dim,
      kernel_size=self.patch_size,
      stride=self.patch_size,
      bias=False,
    )

    self.num_patches = (self.image_size // self.patch_size)**2
    self.num_positions = self.num_patches + 1
    self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

  def __call__(self, x: mx.array) -> mx.array:
    batch_size = x.shape[0]
    patch_embeddings = self.patch_embedding(x)
    patch_embeddings = mx.flatten(patch_embeddings, start_axis=1, end_axis=2)
    embed_dim = patch_embeddings.shape[-1]
    cls_embeddings = mx.broadcast_to(self.class_embedding, (batch_size, 1, embed_dim))
    embeddings = mx.concatenate((cls_embeddings, patch_embeddings), axis=1)
    embeddings += self.position_embedding.weight
    return embeddings


class ClipVisionModel(nn.Module):
  def __init__(self, config: VisionConfig):
    super().__init__()
    self.embeddings = VisionEmbeddings(config)
    self.pre_layrnorm = nn.LayerNorm(config.hidden_size)
    self.encoder = VisionEncoder(config)
    self.post_layernorm = nn.LayerNorm(config.hidden_size)

  def __call__(
    self,
    x: mx.array,
    output_hidden_states: Optional[bool] = None,
  ) -> mx.array:
    x = self.embeddings(x)
    x = self.pre_layrnorm(x)

    encoder_states = (x,) if output_hidden_states else None

    for l in self.encoder.layers:
      x = l(x, mask=None)
      if output_hidden_states:
        encoder_states = encoder_states + (x,)

    pooler_output = self.post_layernorm(x[:, 0, :])
    return pooler_output, x, encoder_states


class VisionModel(nn.Module):
  def __init__(self, config: VisionConfig):
    super().__init__()

    self.model_type = config.model_type
    if self.model_type != "clip_vision_model":
      raise ValueError(f"Unsupported model type: {self.model_type}")

    self.vision_model = ClipVisionModel(config)

  def __call__(self, x: mx.array, output_hidden_states: Optional[bool] = None) -> mx.array:
    return self.vision_model(x, output_hidden_states)

  def sanitize(self, weights):
    sanitized_weights = {}
    for k, v in weights.items():
      if "position_ids" in k:
        # Remove unused position_ids
        continue
      elif "patch_embedding.weight" in k:
        # PyTorch conv2d weight tensors have shape:
        #   [out_channels, in_channels, kH, KW]
        # MLX conv2d expects the weight be of shape:
        #   [out_channels, kH, KW, in_channels]
        sanitized_weights[k] = v.transpose(0, 2, 3, 1)
      else:
        sanitized_weights[k] = v

    return sanitized_weights


@dataclass
class TextConfig:
  model_type: str
  hidden_size: int = 4096
  num_hidden_layers: int = 32
  intermediate_size: int = 11008
  num_attention_heads: int = 32
  head_dim: int = None
  rms_norm_eps: float = 1e-6
  vocab_size: int = 32000
  num_key_value_heads: int = None
  rope_theta: float = 10000
  rope_traditional: bool = False
  rope_scaling: Optional[Dict[str, Union[float, str]]] = None

  @classmethod
  def from_dict(cls, params):
    return cls(**{k: v for k, v in params.items() if k in inspect.signature(cls).parameters})

  def __post_init__(self):
    if self.num_key_value_heads is None:
      self.num_key_value_heads = self.num_attention_heads

    if self.head_dim is None:
      self.head_dim = self.hidden_size // self.num_attention_heads

    if self.model_type is None:
      self.model_type = "llama"

    if self.rope_scaling:
      required_keys = {"factor", "type"}
      if not all(key in self.rope_scaling for key in required_keys):
        raise ValueError(f"rope_scaling must contain keys {required_keys}")

      if self.rope_scaling["type"] != "linear":
        raise ValueError("rope_scaling 'type' currently only supports 'linear'")


class TextAttention(nn.Module):
  def __init__(self, config: TextConfig):
    super().__init__()

    dim = config.hidden_size
    self.n_heads = n_heads = config.num_attention_heads
    self.n_kv_heads = n_kv_heads = config.num_key_value_heads

    self.repeats = n_heads // n_kv_heads

    head_dim = config.hidden_size // n_heads
    self.scale = head_dim**-0.5

    self.q_proj = nn.Linear(dim, n_heads*head_dim, bias=False)
    self.k_proj = nn.Linear(dim, n_kv_heads*head_dim, bias=False)
    self.v_proj = nn.Linear(dim, n_kv_heads*head_dim, bias=False)
    self.o_proj = nn.Linear(n_heads*head_dim, dim, bias=False)

    rope_scale = (1/config.rope_scaling["factor"] if config.rope_scaling is not None and config.rope_scaling["type"] == "linear" else 1)
    self.rope = nn.RoPE(
      head_dim,
      traditional=config.rope_traditional,
      base=config.rope_theta,
      scale=rope_scale,
    )

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


class TextMLP(nn.Module):
  def __init__(self, dim, hidden_dim):
    super().__init__()
    self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
    self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
    self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

  def __call__(self, x) -> mx.array:
    return self.down_proj(nn.silu(self.gate_proj(x))*self.up_proj(x))


class TransformerBlock(nn.Module):
  def __init__(self, config: TextConfig):
    super().__init__()
    self.num_attention_heads = config.num_attention_heads
    self.hidden_size = config.hidden_size
    self.self_attn = TextAttention(config)
    self.mlp = TextMLP(config.hidden_size, config.intermediate_size)
    self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.config = config

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


class Llama(nn.Module):
  def __init__(self, config: TextConfig, shard: Shard):
    super().__init__()
    self.config = config
    self.shard = shard
    self.vocab_size = config.vocab_size
    self.model_type = config.model_type
    self.num_hidden_layers = config.num_hidden_layers
    self.num_key_value_heads = config.num_key_value_heads
    self.head_dim = config.head_dim
    assert self.vocab_size > 0
    if self.shard.is_first_layer():
      self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
    self.layers = []
    for i in range(self.num_hidden_layers):
      if self.shard.start_layer <= i <= self.shard.end_layer:
        self.layers.append(TransformerBlock(config=config))
      else:
        self.layers.append(IdentityBlock())
    if self.shard.is_last_layer():
      self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

  def __call__(
    self,
    inputs: mx.array,
    cache=None,
    inputs_embeds=None,
  ):
    # for passing merged input embeddings
    if inputs_embeds is None:
      if self.shard.is_first_layer():
        h = self.embed_tokens(inputs)
      else:
        h = inputs
    else:
      h = inputs_embeds

    mask = None
    if h.shape[1] > 1:
      mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
      mask = mask.astype(h.dtype)

    if cache is None:
      cache = [None]*len(self.layers)

    for layer, c in zip(self.layers, cache):
      h = layer(h, mask, c)

    if self.shard.is_last_layer():
      h = self.norm(h)
    return h


class LanguageModel(nn.Module):
  def __init__(self, config: TextConfig, shard: Shard):
    super().__init__()
    self.model_type = config.model_type
    if self.model_type != "llama":
      raise ValueError(f"Model type {self.model_type} not supported. Currently only 'llama' is supported")
    self.shard = shard
    self.model = Llama(config, shard)
    if self.shard.is_last_layer():
      self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

  def __call__(
    self,
    inputs: mx.array,
    cache=None,
    inputs_embeds=None,
  ):
    out = self.model(inputs, cache, inputs_embeds)
    if self.shard.is_last_layer():
      out = self.lm_head(out)
    return out

  def sanitize(self, weights):
    shard_state_dict = {}
    for key, value in weights.items():
      if "self_attn.rotary_emb.inv_freq" in key:
        continue

      if key.startswith('language_model.model.layers.'):
        layer_num = int(key.split('.')[3])
        if layer_num < self.shard.start_layer or layer_num > self.shard.end_layer:
          continue
      if not self.shard.is_first_layer() and key.startswith('language_model.model.embed_tokens'):
        continue
      elif not self.shard.is_last_layer() and (key.startswith('language_model.model.norm') or key.startswith('language_model.lm_head')):
        continue

      shard_state_dict[key] = value

    return shard_state_dict


@dataclass
class LlaVAConfig(BaseModelArgs):
  text_config: TextConfig
  vision_config: VisionConfig = None
  model_type: str = "llava"
  ignore_index: int = -100
  image_token_index: int = 32000
  vision_feature_select_strategy: str = "default"
  vision_feature_layer: int = -2
  vocab_size: int = 32000

  @classmethod
  def from_dict(cls, params):
    updated_params = {}
    class_params = inspect.signature(cls).parameters
    for k, v in params.items():
      if k in class_params:
        if k in ["text_config", "vision_config"]:
          v = class_params[k].annotation.from_dict(v)
        updated_params.update({k: v})

    return cls(**updated_params)


@dataclass
class ModelArgs(LlaVAConfig):
  shard: Shard = field(default_factory=lambda: Shard("", 0, 0, 0))

  def __post_init__(self):
    if isinstance(self.shard, dict):
      self.shard = Shard(**self.shard)

    if not isinstance(self.shard, Shard):
      raise TypeError(f"Expected shard to be a Shard instance or a dict, got {type(self.shard)} instead")

    if not self.shard.is_first_layer():
      self.vision_config = None


class LlavaMultiModalProjector(nn.Module):
  def __init__(self, config: LlaVAConfig):
    super().__init__()
    self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
    self.gelu = nn.GELU()
    self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

  def __call__(self, x: mx.array) -> mx.array:
    x = self.linear_1(x)
    x = self.gelu(x)
    x = self.linear_2(x)
    return x


class Model(nn.Module):
  def __init__(self, config: ModelArgs):
    super().__init__()
    self.config = config
    self.model_type = config.model_type
    if config.vision_config:
      self.vision_tower = VisionModel(config.vision_config)
      self.multi_modal_projector = LlavaMultiModalProjector(config)
      self.vision_feature_layer = config.vision_feature_layer
      self.vision_feature_select_strategy = config.vision_feature_select_strategy
    self.language_model = LanguageModel(config.text_config, config.shard)

  def get_input_embeddings(
    self,
    input_ids: Optional[mx.array] = None,
    pixel_values: Optional[mx.array] = None,
  ):
    if pixel_values is None:
      return self.language_model(input_ids)

    # Get the input embeddings from the language model
    inputs_embeds = self.language_model.model.embed_tokens(input_ids)

    # Get the ouptut hidden states from the vision model
    *_, hidden_states = self.vision_tower(pixel_values.transpose(0, 2, 3, 1), output_hidden_states=True)

    # Select the hidden states from the desired layer
    selected_image_feature = hidden_states[self.vision_feature_layer]

    if self.vision_feature_select_strategy == "default":
      selected_image_feature = selected_image_feature[:, 1:]
    elif self.vision_feature_select_strategy == "full":
      selected_image_feature = selected_image_feature
    else:
      raise ValueError("Unexpected feature selection strategy: "
                       f"{self.vision_feature_select_strategy}")

    # Pass image features through the multi-modal projector
    image_features = self.multi_modal_projector(selected_image_feature)

    # Insert special image tokens in the input_ids
    final_inputs_embeds = self._merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids)
    return final_inputs_embeds

  def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids):
    image_token_index = self.config.image_token_index
    num_images, num_image_patches, embed_dim = image_features.shape

    # Positions of <image> tokens in input_ids, assuming batch size is 1
    image_positions = np.where(input_ids[0] == image_token_index)[0].tolist()

    if len(image_positions) != num_images:
      raise ValueError(f"The number of image tokens ({len(image_positions)}) does not "
                       f" match the number of image inputs ({num_images}).")

    text_segments = []
    start_idx = 0

    for position in image_positions:
      text_segments.append(inputs_embeds[:, start_idx:position])
      start_idx = position + 1

    image_embeddings = mx.split(image_features, image_features.shape[0])
    final_embeddings = [v for p in zip(text_segments, image_embeddings) for v in p]
    final_embeddings += [inputs_embeds[:, start_idx:]]

    # Create a final embedding of shape
    # (1, num_image_patches*num_images + sequence_len, embed_dim)
    return mx.concatenate(final_embeddings, axis=1)

  def __call__(self, input_ids: mx.array, pixel_values: mx.array = None, cache=None):
    input_embddings = None
    if pixel_values is not None:
      input_embddings = self.get_input_embeddings(input_ids, pixel_values)
    logits = self.language_model(input_ids, cache=cache, inputs_embeds=input_embddings)
    return logits

  def sanitize(self, weights):
    if self.config.vision_config:
      weights = self.vision_tower.sanitize(weights)
    else:
      weights = {k: v for k, v in weights.items() if not k.startswith(('vision_tower', 'multi_modal_projector', 'vision_feature_layer', 'vision_feature_select_strategy'))}
    weights = self.language_model.sanitize(weights)
    return weights

  @property
  def layers(self):
    return self.language_model.model.layers

  @property
  def head_dim(self):
    return (self.language_model.model.head_dim or self.language_model.model.hidden_size // self.language_model.model.num_attention_heads)

  @property
  def n_kv_heads(self):
    return self.language_model.model.num_key_value_heads
