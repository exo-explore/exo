import math
import inspect
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field

from ...shard import Shard
from .base import IdentityBlock

import mlx.nn as nn
import mlx.core as mx

from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.llama import initialize_rope

@dataclass
class MllamaVisionConfig:
    hidden_size: int = 1280
    num_hidden_layers: int = 32
    num_global_layers: int = 8
    num_attention_heads: int = 16
    num_channels: int = 3
    intermediate_size: int = 5120
    vision_output_dim: int = 7680
    image_size: int = 448
    patch_size: int = 14
    norm_eps: float = 1e-5
    max_num_tiles: int = 4
    intermediate_layers_indices: Optional[List[int]] = None
    supported_aspect_ratios: Optional[List[List[int]]] = None
    initializer_range: float = 0.02

    def __post_init__(self):
        if self.supported_aspect_ratios is None:
            if self.max_num_tiles != 4:
                raise ValueError("max_num_tiles must be 4 for default supported aspect ratios")
            self.supported_aspect_ratios = [[1, 1], [1, 2], [1, 3], [1, 4], [2, 1], [2, 2], [3, 1], [4, 1]]

        if self.intermediate_layers_indices is None:
            self.intermediate_layers_indices = [3, 7, 15, 23, 30]
            
        self.max_aspect_ratio_id = len(self.supported_aspect_ratios)
        
    @classmethod
    def from_dict(cls, params):
        updated_params = {}
        class_params = inspect.signature(cls).parameters
        for k, v in params.items():
            if k in class_params:
                updated_params.update({k: v})
        return cls(**updated_params)


@dataclass
class MllamaTextConfig:
    vocab_size: int = 128256
    hidden_size: int = 4096
    head_dim: int = None
    num_hidden_layers: int = 40
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    intermediate_size: int = 14_336
    rope_theta: float = 500000
    rope_scaling: Optional[Dict] = None
    rope_traditional: bool = False
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 131072
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False
    cross_attention_layers: Optional[List[int]] = None
    shard: Optional[Shard] = None
    
    def __post_init__(self):
        if self.cross_attention_layers is None:
            self.cross_attention_layers = [3, 8, 13, 18, 23, 28, 33, 38]
        self.head_dim = self.hidden_size // self.num_attention_heads
            
    @classmethod
    def from_dict(cls, params):
        updated_params = {}
        class_params = inspect.signature(cls).parameters
        for k, v in params.items():
            if k in class_params:
                updated_params.update({k: v})
        return cls(**updated_params)

@dataclass
class MllamaConfig(BaseModelArgs):
    text_config: MllamaTextConfig
    vision_config: MllamaVisionConfig = None
    image_token_index: int = 128256

    def __post_init__(self):
        if self.vision_config is None:
            self.vision_config = MllamaVisionConfig()
        elif isinstance(self.vision_config, dict):
            self.vision_config = MllamaVisionConfig(**self.vision_config)
        elif isinstance(self.vision_config, MllamaVisionConfig):
            self.vision_config = self.vision_config

        if self.text_config is None:
            self.text_config = MllamaTextConfig()
        elif isinstance(self.text_config, dict):
            self.text_config = MllamaTextConfig(**self.text_config)
        elif isinstance(self.text_config, MllamaTextConfig):
            self.text_config = self.text_config
            
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
class ModelArgs(MllamaConfig):
  shard: Shard = field(default_factory=lambda: Shard("", 0, 0, 0))

  def __post_init__(self):
    super().__post_init__()
    if isinstance(self.shard, dict):
      self.shard = Shard(**self.shard)

    if not isinstance(self.shard, Shard):
      raise TypeError(f"Expected shard to be a Shard instance or a dict, got {type(self.shard)} instead")

    if not self.shard.is_first_layer():
      self.vision_config = None
      
    self.text_config.shard = self.shard


def _prepare_aspect_ratio_attention_mask(
    aspect_ratio_mask: mx.array,
    num_patches: int,
    target_length: int,
) -> mx.array:
    # Expand aspect ratio mask to target_length
    batch_size, max_num_tiles = aspect_ratio_mask.shape
    attention_mask = aspect_ratio_mask.reshape(batch_size, max_num_tiles, 1, 1)
    attention_mask = mx.tile(attention_mask, (1, 1, target_length, 1))

    # Mask padding patches
    pad_patches = target_length - num_patches
    attention_mask[:, :, -pad_patches:] = 0

    # Invert the mask (0 -> 1, 1 -> 0)
    attention_mask = 1 - attention_mask

    # Reshape to 2D and create 4D attention mask
    # (batch_size, 1, max_num_tiles * target_length, max_num_tiles * target_length)
    attention_mask = attention_mask.reshape(batch_size, max_num_tiles * target_length, 1)
    attention_mask = attention_mask @ attention_mask.transpose(0, -1, -2) * -3.3895313892515355e+38
    attention_mask = mx.expand_dims(attention_mask, axis=1)

    return attention_mask

class MllamaPrecomputedAspectRatioEmbedding(nn.Module):
    def __init__(self, config: MllamaVisionConfig, is_gated: bool = True):
        super().__init__()
        self.max_num_tiles = config.max_num_tiles
        self.hidden_size = config.hidden_size
        self.max_aspect_ratio_id = config.max_aspect_ratio_id
        self.is_gated = is_gated

        self.embedding = nn.Embedding(self.max_aspect_ratio_id + 1, self.max_num_tiles * self.hidden_size)
        if is_gated:
            self.gate = mx.zeros(1)

    def __call__(self, hidden_state: mx.array, aspect_ratio_ids: mx.array) -> mx.array:
        embeddings = self.embedding(aspect_ratio_ids)
        embeddings = embeddings.reshape(-1, self.max_num_tiles, 1, self.hidden_size)

        if self.is_gated:
            embeddings = embeddings * mx.tanh(self.gate)

        hidden_state = hidden_state + embeddings
        return hidden_state


class MllamaPrecomputedPositionEmbedding(nn.Module):
    def __init__(self, config: MllamaVisionConfig):
        super().__init__()
        self.max_num_tiles = config.max_num_tiles
        self.max_aspect_ratio_id = config.max_aspect_ratio_id
        self.num_patches = (config.image_size // config.patch_size) ** 2 + 1
        self.hidden_size = config.hidden_size

        self.gate = mx.zeros(1)

        # position embedding
        position_embedding = mx.random.uniform(0, 1, shape=(self.num_patches, self.hidden_size))
        self.embedding =  self.hidden_size**-0.5 * position_embedding

        # tile position embedding
        self.tile_embedding = nn.Embedding(
            self.max_aspect_ratio_id + 1, self.max_num_tiles * self.num_patches * self.hidden_size
        )

    def __call__(self, hidden_state: mx.array, aspect_ratio_ids: mx.array) -> mx.array:
        # position embeddings
        gated_position_embedding = (1 - mx.tanh(self.gate)) * self.embedding
        hidden_state = hidden_state + gated_position_embedding.reshape(1, 1, self.num_patches, self.hidden_size)

        # precomputed tile position embeddings
        tile_position_embedding = self.tile_embedding(aspect_ratio_ids)
        batch_size = hidden_state.shape[0]
        tile_position_embedding = tile_position_embedding.reshape(
            batch_size, self.max_num_tiles, self.num_patches, self.hidden_size
        )
        gated_tile_position_embedding = mx.tanh(self.gate) * tile_position_embedding
        hidden_state = hidden_state + gated_tile_position_embedding

        return hidden_state
    
    
class MllamaVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(hidden_states)))


class MllamaVisionAttention(nn.Module):
    def __init__(self, config: MllamaVisionConfig):
        super().__init__()

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.embed_dim, bias=False)

    def __call__(
        self,
        hidden_state: mx.array,
        attention_mask: Optional[mx.array] = None
    ) -> mx.array:
        query = self.q_proj(hidden_state)
        key = self.k_proj(hidden_state)
        value = self.v_proj(hidden_state)

        batch_size, q_seq_len, _ = query.shape
        _, kv_seq_len, _ = key.shape

        query = query.reshape(batch_size, q_seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        key = key.reshape(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        value = value.reshape(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        attn_output = mx.fast.scaled_dot_product_attention(query, key, value, scale=self.scale, mask=attention_mask)

        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, q_seq_len, -1)

        output = self.o_proj(attn_output)

        return output
    
    
class MllamaVisionEncoderLayer(nn.Module):
    def __init__(self, config: MllamaVisionConfig, is_gated: bool = False):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.is_gated = is_gated
        self.intermediate_size = config.intermediate_size

        self.self_attn = MllamaVisionAttention(config)
        self.mlp = MllamaVisionMLP(config)

        self.input_layernorm = nn.LayerNorm(self.hidden_size, eps=config.norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(self.hidden_size, eps=config.norm_eps)

        if is_gated:
            self.gate_attn = mx.ones(1) * math.pi / 4
            self.gate_ffn = mx.ones(1) * math.pi / 4

    def __call__(
        self,
        hidden_state: mx.array,
        attention_mask: Optional[mx.array] = None
    ):
        residual = hidden_state
        hidden_state = self.input_layernorm(hidden_state)
        hidden_state = self.self_attn(hidden_state, attention_mask=attention_mask)
        if self.is_gated:
            hidden_state = mx.tanh(self.gate_attn) * hidden_state
        hidden_state = residual + hidden_state

        # Feed forward
        residual = hidden_state
        hidden_state = self.post_attention_layernorm(hidden_state)
        hidden_state = self.mlp(hidden_state)
        if self.is_gated:
            hidden_state = mx.tanh(self.gate_ffn) * hidden_state
        return residual + hidden_state


class MllamaVisionEncoder(nn.Module):
    def __init__(self, config: MllamaVisionConfig, num_layers=32, is_gated=False):
        super().__init__()
        self.config = config
        self.layers = [MllamaVisionEncoderLayer(config, is_gated) for _ in range(num_layers)]

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None) -> mx.array:
        int_hidden_states = ()
        for layer in self.layers:
            int_hidden_states += (hidden_states,)
            hidden_states = layer(
                hidden_state=hidden_states,
                attention_mask=attention_mask
            )
        int_hidden_states += (hidden_states,)
        return hidden_states, int_hidden_states
    
    
class MllamaTextCrossAttention(nn.Module):
    def __init__(
        self,
        config: Optional[MllamaTextConfig] = None
    ):
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        cross_attention_states: Optional[mx.array] = None,
        cache = None
    ) -> mx.array:
        bsz, q_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states)
        query_states = query_states.reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        query_states = self.q_norm(query_states)

        if cross_attention_states is not None:
            key_states = self.k_proj(cross_attention_states)
            value_states = self.v_proj(cross_attention_states)
            key_states = key_states.reshape(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
            value_states = value_states.reshape(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)

            if cache is not None:
                # if we have a new image + new tokens, we only computed key_states on that new image
                # we still update the cross key states, past_image, new_image. And use it!
                key_states, value_states = cache.update_and_fetch(key_states, value_states)
        else:
            key_states, value_states = cache.state

        key_states = self.k_norm(key_states)
        attn_output = mx.fast.scaled_dot_product_attention(query_states, key_states, value_states, scale=self.scale)

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(bsz, q_len, -1)
        return self.o_proj(attn_output)

class MllamaTextSelfAttention(nn.Module):
    def __init__(self, config: MllamaTextConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.rope = initialize_rope(config)

    def __call__(
        self,
        hidden_states: mx.array,
        cache = None
    ):
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        key_states = key_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        value_states = value_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            query_states = self.rope(query_states, offset=cache.offset)
            key_states = self.rope(key_states, offset=cache.offset)
            key_states, value_states = cache.update_and_fetch(key_states, value_states)
        else:
            query_states = self.rope(query_states)
            key_states = self.rope(key_states)

        attn_output = mx.fast.scaled_dot_product_attention(query_states, key_states, value_states, scale=self.scale).transpose(0, 2, 1, 3).reshape(bsz, q_len, -1)
        return self.o_proj(attn_output)
    

class MllamaTextMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class MllamaSelfAttentionDecoderLayer(nn.Module):
    def __init__(self, config: MllamaTextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = MllamaTextSelfAttention(config=config)

        self.mlp = MllamaTextMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        cache = None,
        **_
    ) -> mx.array:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states, cache=cache)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class MllamaCrossAttentionDecoderLayer(nn.Module):
    def __init__(self, config: MllamaTextConfig) -> None:
        self.cross_attn = MllamaTextCrossAttention(config)

        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cross_attn_attn_gate = mx.zeros(1)

        self.mlp = MllamaTextMLP(config)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cross_attn_mlp_gate = mx.zeros(1)

    def __call__(
        self,
        hidden_states: mx.array,
        cross_attention_states: mx.array,
        cache = None,
        **_
    ) -> Tuple[mx.array]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.cross_attn(
            hidden_states=hidden_states,
            cross_attention_states=cross_attention_states,
            cache=cache
        )
        hidden_states = residual + mx.tanh(self.cross_attn_attn_gate) * hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + mx.tanh(self.cross_attn_mlp_gate) * hidden_states
        return hidden_states
    

class MllamaVisionModel(nn.Module):
    def __init__(self, config: MllamaVisionConfig):
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.max_num_tiles = config.max_num_tiles
        self.hidden_size = config.hidden_size
        self.num_channels = config.num_channels
        self.intermediate_layers_indices = config.intermediate_layers_indices
        self.num_patches = (self.image_size // self.patch_size) ** 2 + 1
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.class_embedding = self.hidden_size**-0.5 * mx.random.uniform(shape=(self.hidden_size,))
        self.gated_positional_embedding = MllamaPrecomputedPositionEmbedding(config)

        self.pre_tile_positional_embedding = MllamaPrecomputedAspectRatioEmbedding(config, is_gated=True)
        self.post_tile_positional_embedding = MllamaPrecomputedAspectRatioEmbedding(config, is_gated=True)

        self.layernorm_pre = nn.LayerNorm(self.hidden_size)
        self.layernorm_post = nn.LayerNorm(self.hidden_size)

        self.transformer = MllamaVisionEncoder(config, config.num_hidden_layers, is_gated=False)
        self.global_transformer = MllamaVisionEncoder(config, config.num_global_layers, is_gated=True)

    def __call__(
        self,
        pixel_values: mx.array,
        aspect_ratio_ids: mx.array,
        aspect_ratio_mask: mx.array
    ) -> mx.array:
        batch_size, num_concurrent_media, num_tiles, num_channels, height, width = pixel_values.shape

        pixel_values = pixel_values.reshape(batch_size * num_concurrent_media * num_tiles, num_channels, height, width).transpose(0, 2, 3, 1)
        aspect_ratio_ids = aspect_ratio_ids.reshape(batch_size * num_concurrent_media, -1)

        # Patch embedding
        patch_embeds = self.patch_embedding(pixel_values)
        hidden_state = patch_embeds.flatten(1, 2)
        
        # Tile embeddings
        _, num_patches, dim = hidden_state.shape
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media, num_tiles, -1, dim)
        hidden_state = self.pre_tile_positional_embedding(hidden_state, aspect_ratio_ids)

        # Add cls token
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media * num_tiles, num_patches, dim)
        _batch_size, _, _ = hidden_state.shape
        class_embedding = mx.repeat(mx.expand_dims(self.class_embedding, axis=(0,1)), _batch_size, axis=0)
        hidden_state = mx.concatenate([class_embedding, hidden_state], axis=1)
        num_patches += 1

        # Position embeddings
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media, num_tiles, num_patches, dim)
        hidden_state = self.gated_positional_embedding(hidden_state, aspect_ratio_ids)
        hidden_state = self.layernorm_pre(hidden_state)

        # Compute the number of tokens to pad
        num_padding_patches = (8 - (hidden_state.shape[-2] % 8)) % 8
        # Compute padding tuple for pad function
        padding = [(0, 0) ,(0, 0), (0, num_padding_patches), (0, 0)]  # (pad_left, pad_right, pad_left for dim -2, pad_right for dim -2)
        # Pad the tensor
        hidden_state = mx.pad(hidden_state, pad_width=padding)
        slice_index = -num_padding_patches if num_padding_patches > 0 else None

        # Prepare attention mask
        attention_mask = aspect_ratio_mask.reshape(batch_size * num_concurrent_media, -1).astype(mx.bfloat16)
        attention_mask = _prepare_aspect_ratio_attention_mask(
            aspect_ratio_mask=attention_mask,
            num_patches=self.num_patches,
            target_length=hidden_state.shape[2]
        )

        # Apply encoder
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media, -1, dim)
        hidden_state, int_hidden_states = self.transformer(
            hidden_state,
            attention_mask=attention_mask
        )
        hidden_state = self.layernorm_post(hidden_state)

        # Apply global encoder
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media, num_tiles, num_patches + num_padding_patches, dim
        )
        hidden_state = self.post_tile_positional_embedding(hidden_state, aspect_ratio_ids)
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media, num_tiles * (num_patches + num_padding_patches), dim
        )
        hidden_state, _ = self.global_transformer(
            hidden_state,
            attention_mask=attention_mask
        )
        # Remove padding form hidden state
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media, num_tiles, num_patches + num_padding_patches, dim
        )
        hidden_state = hidden_state[:, :, :slice_index]
        hidden_state = hidden_state.reshape(batch_size, num_concurrent_media, num_tiles, num_patches, dim)

        # Collect intermediate layer outputs from encoder output
        intermediate_hidden_states = mx.stack(int_hidden_states, axis=-1)
        intermediate_hidden_states = intermediate_hidden_states[..., self.intermediate_layers_indices]

        # Remove padding from intermediate hidden states
        intermediate_hidden_states = intermediate_hidden_states.reshape(
            batch_size * num_concurrent_media, num_tiles, num_patches + num_padding_patches, -1
        )
        intermediate_hidden_states = intermediate_hidden_states[:, :, :slice_index]
        intermediate_hidden_states = intermediate_hidden_states.reshape(
            batch_size, num_concurrent_media, num_tiles, num_patches, -1
        )

        # Concatenate final hidden state and intermediate hidden states
        hidden_state = mx.concatenate([hidden_state, intermediate_hidden_states], axis=-1)
        return hidden_state
    
    def sanitize(self, weights):
        sanitized_weights = {}

        for key, value in weights.items():
            if "patch_embedding" in key:
                sanitized_weights[key] = value.transpose(0, 2, 3, 1)
            else:
                sanitized_weights[key] = value
        return sanitized_weights


class MllamaTextModel(nn.Module):
    def __init__(self, config: MllamaTextConfig):
        self.vocab_size = config.vocab_size
        self.cross_attention_layers = config.cross_attention_layers
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.shard = config.shard
        if self.shard.is_first_layer():
            self.embed_tokens = nn.Embedding(config.vocab_size + 8, config.hidden_size)
        self.layers = []
        for layer_idx in range(config.num_hidden_layers):
            if self.shard.start_layer > layer_idx or layer_idx > self.shard.end_layer:
                self.layers.append(IdentityBlock())
            elif layer_idx in self.cross_attention_layers:
                self.layers.append(MllamaCrossAttentionDecoderLayer(config))
            else:
                self.layers.append(MllamaSelfAttentionDecoderLayer(config))
        if self.shard.is_last_layer():
            self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        cross_attention_states: Optional[mx.array] = None,
        cache = None
    ) -> mx.array:
        if self.shard.is_first_layer():
            hidden_states = self.embed_tokens(inputs)
        else:
            hidden_states = inputs
        for idx, (decoder_layer, c) in enumerate(zip(self.layers, cache)):
            # For text-only path we should skip cross attention layers.
            # Let's check if the layer is cross attention layer and if we have cross attention states
            # or cached cross attention states.
            if (idx in self.cross_attention_layers) and cross_attention_states is None and c.keys is None:
                continue
            
            hidden_states = decoder_layer(
                hidden_states,
                cross_attention_states=cross_attention_states,
                cache=c
            )
        if self.shard.is_last_layer():
            return self.lm_head(self.norm(hidden_states))
        else:
            return hidden_states
    
    def sanitize(self, weights):
        shard_state_dict = {}
        for key, value in weights.items():
            key = key.replace("language_model.model.", "language_model.")
            if key.startswith('language_model.layers.'):
                layer_num = int(key.split('.')[2])
                if layer_num < self.shard.start_layer or layer_num > self.shard.end_layer:
                    continue
            if not self.shard.is_first_layer() and key.startswith('language_model.embed_tokens'):
                continue
            elif not self.shard.is_last_layer() and (key.startswith('language_model.norm') or key.startswith('language_model.lm_head')):
                continue

            shard_state_dict[key] = value

        return shard_state_dict

    
    
class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        self.vocab_size = config.text_config.vocab_size
        self.hidden_size = config.text_config.hidden_size
        self.language_model = MllamaTextModel(config.text_config)
        self.shard = config.shard
        self.vision_model = None
        if config.vision_config:
            self.vision_model = MllamaVisionModel(config.vision_config)
            self.multi_modal_projector = nn.Linear(
                config.vision_config.vision_output_dim,
                config.text_config.hidden_size,
                bias=True,
            )
        self.model_type = "mllama"

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        aspect_ratio_mask: Optional[mx.array] = None,
        aspect_ratio_ids: Optional[mx.array] = None,
        cache = None,
        inference_state: Optional[mx.array] = None
    ) -> mx.array:
        cross_attention_states = inference_state
        if pixel_values is not None:
            if aspect_ratio_ids is None:
                raise ValueError("`aspect_ratio_ids` must be provided if `pixel_values` is provided")
            # get vision tokens from vision model
            cross_attention_states = self.vision_model(
                pixel_values=pixel_values,
                aspect_ratio_ids=aspect_ratio_ids,
                aspect_ratio_mask=aspect_ratio_mask
            )
            cross_attention_states = self.multi_modal_projector(cross_attention_states).reshape(
                -1, cross_attention_states.shape[-2], self.hidden_size
            )
        outputs = self.language_model(
            input_ids,
            cross_attention_states=cross_attention_states,
            cache=cache
        )
        if self.shard.is_last_layer():
            cross_attention_states = None
        return outputs, cross_attention_states
    
    def sanitize(self, weights):
        if self.vision_model:
            weights = self.vision_model.sanitize(weights)
        else:
            weights = {k: v for k, v in weights.items() if not k.startswith(('vision_model', 'multi_modal_projector'))}
        weights = self.language_model.sanitize(weights)
        return weights

    @property
    def layers(self):
        return self.language_model.layers

    @property
    def head_dim(self):
        return (self.language_model.head_dim or self.language_model.hidden_size // self.language_model.num_attention_heads)

    @property
    def n_kv_heads(self):
        return self.language_model.num_key_value_heads