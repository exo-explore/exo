import inspect
from dataclasses import dataclass, field
from typing import Optional, List

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_lm.models.base import BaseModelArgs

from exo.inference.shard import Shard
from exo.inference.mlx.models.llama import LlamaModel, ModelArgs as LlamaModelArgs
from exo.inference.mlx.models.llava import LlavaMultiModalProjector


@dataclass
class PixtralVisionConfig:
  model_type: str
  num_hidden_layers: int = 24
  hidden_size: int = 1024
  intermediate_size: int = 4096
  num_attention_heads: int = 16
  image_size: int = 1024
  patch_size: int = 16
  projection_dim: int = 768
  vocab_size: int = 32000
  num_channels: int = 3
  rope_theta: float = 10000.0
  tie_word_embeddings: bool = False

  @classmethod
  def from_dict(cls, params):
    return cls(**{k: v for k, v in params.items() if k in inspect.signature(cls).parameters})

  def __post_init__(self):
    self.head_dim = self.hidden_size // self.num_attention_heads
    
def position_ids_in_meshgrid(patch_embeds_list, max_width):
    positions = []
    for patch in patch_embeds_list:
        height, width = patch.shape[1:3]
        mesh = mx.meshgrid(mx.arange(height), mx.arange(width), indexing="ij")
        h_grid, v_grid = mesh[0].reshape(-1), mesh[1].reshape(-1)
        ids = h_grid * max_width + v_grid
        positions.append(ids)
    return mx.concatenate(positions)


class PixtralRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        rope_theta = config.rope_theta
        dim = config.head_dim
        max_patches_per_side = config.image_size // config.patch_size
        freqs = 1.0 / (rope_theta ** (mx.arange(0, dim, 2) / dim))

        h = mx.arange(max_patches_per_side)
        w = mx.arange(max_patches_per_side)

        freqs_h = mx.outer(h, freqs[::2])
        freqs_w = mx.outer(w, freqs[1::2])
        inv_freq = mx.concatenate(
            [
                mx.repeat(freqs_h[:, None, :], repeats=max_patches_per_side, axis=1),
                mx.repeat(freqs_w[None, :, :], repeats=max_patches_per_side, axis=0),
            ],
            axis=-1,
        ).reshape(-1, dim // 2)
        self._inv_freq = mx.concatenate((inv_freq, inv_freq), axis=-1)

    def __call__(self, _, position_ids):
        freqs = self._inv_freq[position_ids]
        return mx.cos(freqs), mx.sin(freqs)


def rotate_half(x: mx.array) -> mx.array:
    x1, x2 = mx.split(x, 2, axis=-1)
    return mx.concatenate([-x2, x1], axis=-1)

def apply_rotary_pos_emb(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array, axis: int = 1):
    cos = mx.expand_dims(cos, axis=axis)
    sin = mx.expand_dims(sin, axis=axis)
    
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    
    return q_embed, k_embed


class PixtralAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.scale = self.head_dim**-0.5

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    def __call__(
        self,
        h: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_embeddings: Optional[mx.array] = None,
    ):

        B, L, _ = h.shape

        queries = self.q_proj(h)
        keys = self.k_proj(h)
        values = self.v_proj(h)
        
        queries = queries.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        if position_embeddings is not None:
            cos, sin = position_embeddings
            queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin, axis=0)
        
        scores = (queries*self.scale) @ keys.transpose(0, 1, 3, 2)
        
        if attention_mask is not None:
            scores = scores + attention_mask
        
        scores = mx.softmax(scores, axis=-1)
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(values_hat)


class PixtralMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.gelu = nn.GELU()

    def __call__(self, h):
        return self.down_proj(self.gelu(self.gate_proj(h)) * self.up_proj(h))


class PixtralRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps

    def __call__(self, h: mx.array):
        v = mx.mean(h ** 2, axis=-1, keepdims=True)
        h = h * mx.rsqrt(v + self.eps)
        return self.weight * h


class PixtralAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = PixtralRMSNorm(config.hidden_size, eps=1e-5)
        self.feed_forward = PixtralMLP(config)
        self.attention = PixtralAttention(config)
        self.ffn_norm = PixtralRMSNorm(config.hidden_size, eps=1e-5)

    def __call__(
        self,
        h: mx.array,
        attention_mask: mx.array,
        position_embeddings: Optional[mx.array] = None,
    ):
        r = h
        h = r + self.attention(self.attention_norm(h), attention_mask, position_embeddings)

        r = h
        h = r + self.feed_forward(self.ffn_norm(h))

        return h


class PixtralTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = []
        for _ in range(config.num_hidden_layers):
            self.layers.append(PixtralAttentionLayer(config=config))

    def __call__(
        self,
        inputs_embeds,
        attention_mask: Optional[mx.array] = None,
        position_embeddings: Optional[mx.array] = None,
    ):
        h = inputs_embeds
        for layer in self.layers:
            h = layer(
                h,
                attention_mask,
                position_embeddings
            )
            
        return h

def generate_block_attention_mask(patch_embeds_list, input_array):
    seq_len = input_array.shape[1]
    dmin = np.finfo(np.array(input_array).dtype).min
    causal_mask = mx.full((seq_len, seq_len), dmin, dtype=input_array.dtype)

    block_end_idx = mx.cumsum(mx.array(patch_embeds_list), axis=-1).tolist()
    block_start_idx = mx.cumsum(mx.array([0] + patch_embeds_list[:-1]), axis=-1).tolist()
    
    for start, end in zip(block_start_idx, block_end_idx):
        causal_mask[start:end, start:end] = 0

    causal_mask = mx.broadcast_to(causal_mask[None, None, :, :], (input_array.shape[0], 1, seq_len, seq_len))
    return causal_mask


class PixtralModel(nn.Module):
    def __init__(self, config: PixtralVisionConfig):
        super().__init__()
        self.config = config
        self.patch_conv = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        self.ln_pre = PixtralRMSNorm(config.hidden_size, eps=1e-5)
        self.transformer = PixtralTransformer(config)
        self.patch_positional_embedding = PixtralRotaryEmbedding(config)

    def __call__(self, pixel_values: List[mx.array]):
        patch_embeds_list = [self.patch_conv(mx.expand_dims(img, axis=0).transpose(0, 2, 3, 1)) for img in pixel_values]

        patch_embeds = mx.concatenate([p.flatten(1, 2) for p in patch_embeds_list], axis=1)
        patch_embeds = self.ln_pre(patch_embeds)

        position_ids = position_ids_in_meshgrid(
            patch_embeds_list, max_width=self.config.image_size // self.config.patch_size
        )

        position_embedding = self.patch_positional_embedding(patch_embeds, position_ids)
        attention_mask = generate_block_attention_mask(
            [p.shape[1] * p.shape[2] for p in patch_embeds_list], patch_embeds
        )
        return self.transformer(patch_embeds, attention_mask, position_embedding)
    
    def sanitize(self, weights):
        sanitized_weights = {}

        for key, value in weights.items():
            if "patch_conv" in key:
                sanitized_weights[key] = value.transpose(0, 2, 3, 1)
            else:
                sanitized_weights[key] = value
        return sanitized_weights
    
class LanguageModel(nn.Module):
  def __init__(self, config: LlamaModelArgs):
    super().__init__()
    self.model_type = config.model_type
    self.shard = config.shard
    self.model = LlamaModel(config)
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
class PixtralConfig(BaseModelArgs):
  text_config: LlamaModelArgs
  vision_config: PixtralVisionConfig = None
  model_type: str = "pixtral"
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
class ModelArgs(PixtralConfig):
  shard: Shard = field(default_factory=lambda: Shard("", 0, 0, 0))

  def __post_init__(self):
    if isinstance(self.shard, dict):
      self.shard = Shard(**self.shard)

    if not isinstance(self.shard, Shard):
      raise TypeError(f"Expected shard to be a Shard instance or a dict, got {type(self.shard)} instead")

    if not self.shard.is_first_layer():
      self.vision_config = None
      
    self.text_config.shard = self.shard


class Model(nn.Module):
  def __init__(self, config: ModelArgs):
    super().__init__()
    self.config = config
    self.model_type = config.model_type

    if config.vision_config:
      self.vision_tower = PixtralModel(config.vision_config)
      self.multi_modal_projector = LlavaMultiModalProjector(config)
      self.vision_feature_layer = config.vision_feature_layer
      self.vision_feature_select_strategy = config.vision_feature_select_strategy
    self.language_model = LanguageModel(config.text_config)

  def get_input_embeddings(
    self,
    input_ids: Optional[mx.array] = None,
    pixel_values: Optional[mx.array] = None,
  ):
    if pixel_values is None:
      return self.language_model(input_ids)
    inputs_embeds = self.language_model.model.embed_tokens(input_ids)
    hidden_states = self.vision_tower(pixel_values)
    selected_image_feature = hidden_states[self.vision_feature_layer]

    if self.vision_feature_select_strategy == "default":
      selected_image_feature = selected_image_feature[:, 1:]
    elif self.vision_feature_select_strategy == "full":
      selected_image_feature = selected_image_feature
    else:
      raise ValueError("Unexpected feature selection strategy: "
                       f"{self.vision_feature_select_strategy}")

    image_features = mx.expand_dims(self.multi_modal_projector(selected_image_feature), axis=0)

    
    special_image_mask = mx.expand_dims(input_ids == self.config.image_token_index, axis=-1)
    special_image_mask = mx.broadcast_to(special_image_mask, inputs_embeds.shape)
    inputs_embeds = self.masked_scatter(inputs_embeds, special_image_mask, image_features)
    return inputs_embeds

  def masked_scatter(self, inputs_embeds, special_image_mask, image_features):
    flat_result = inputs_embeds.reshape(-1)
    flat_mask = special_image_mask.reshape(-1)
    flat_source = image_features.reshape(-1)
    
    indices = mx.array(np.flatnonzero(flat_mask))
    num_masked = indices.size
    if flat_source.size < num_masked:
        raise Exception("Number of elements of source < number of ones in mask")
    
    flat_result[indices] = flat_source[:num_masked]
    return flat_result.reshape(inputs_embeds.shape)


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