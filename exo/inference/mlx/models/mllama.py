import math
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

import mlx.nn as nn
import mlx.core as mx

from mlx_lm.models.base import BaseModelArgs, KVCache, RotatingKVCache
from mlx_lm.models.llama import initialize_rope

@dataclass
class MllamaVisionConfig:
    hidden_size: int = 1280
    hidden_act: str = "gelu"
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
            
        self.max_aspect_ratio_id = self.supported_aspect_ratios


@dataclass
class MllamaTextConfig:
    vocab_size: int = 128256
    hidden_size: int = 4096
    hidden_act: str = "silu"
    num_hidden_layers: int = 40
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    intermediate_size: int = 14_336
    rope_theta: float = 500_000
    rope_scaling: Optional[Dict] = None
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 131072
    initializer_range: float = 0.02
    use_cache: bool = True
    tie_word_embeddings: bool = False
    cross_attention_layers: Optional[List[int]] = None
    dropout: float = 0
    bos_token_id: int = 128000
    eos_token_id: int = 128001
    pad_token_id: Optional[int] = 128004
    
    def __post_init__(self):
        if self.cross_attention_layers is None:
            self.cross_attention_layers = [3, 8, 13, 18, 23, 28, 33, 38]

@dataclass
class MllamaConfig(BaseModelArgs):
    vision_config: MllamaVisionConfig = None
    text_config: MllamaTextConfig
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


def _prepare_aspect_ratio_attention_mask(
    aspect_ratio_mask: mx.array,
    num_patches: int,
    target_length: int,
) -> mx.array:
    # Expand aspect ratio mask to target_length
    batch_size, max_num_tiles = aspect_ratio_mask.shape
    attention_mask = aspect_ratio_mask.view(batch_size, max_num_tiles, 1, 1)
    attention_mask = attention_mask.tile(1, 1, target_length, 1)

    # Mask padding patches
    pad_patches = target_length - num_patches
    attention_mask[:, :, -pad_patches:] = 0

    # Invert the mask (0 -> 1, 1 -> 0)
    attention_mask = 1 - attention_mask

    # Reshape to 2D and create 4D attention mask
    # (batch_size, 1, max_num_tiles * target_length, max_num_tiles * target_length)
    attention_mask = attention_mask.reshape(batch_size, max_num_tiles * target_length, 1)
    attention_mask = attention_mask @ attention_mask.transpose(0, 1, 3, 2) * -1e9
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
        self.scale = config.hidden_size**-0.5

        self.gate = mx.zeros(1)

        # position embedding
        position_embedding = mx.random.uniform(0, 1, shape=(self.num_patches, self.hidden_size))
        self.embedding = self.scale * position_embedding

        # tile position embedding
        self.tile_embedding = nn.Embedding(
            self.max_aspect_ratio_id + 1, self.max_num_tiles * self.num_patches * self.hidden_size
        )

    def __call__(self, hidden_state: mx.array, aspect_ratio_ids: mx.array) -> mx.array:
        # position embeddings
        gated_position_embedding = (1 - mx.tanh(self.gate)) * self.embedding
        hidden_state = hidden_state + gated_position_embedding.view(1, 1, self.num_patches, self.hidden_size)

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
        self.num_heads = config.attention_heads
        self.head_dim = config.hidden_size // config.attention_heads

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

        query = query.view(batch_size, q_seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, kv_seq_len, self.num_heads, self.head_dim)
        value = value.view(batch_size, kv_seq_len, self.num_heads, self.head_dim)

        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)

        attn_output = query.scaled_dot_product_attention(key, value, mask=attention_mask)

        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, q_seq_len, -1)

        output = self.o_proj(attn_output)

        return output
    
    
class MllamaVisionEncoderLayer(nn.Module):
    def __init__(self, config: MllamaVisionConfig, is_gated: bool = False):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.attention_heads
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
        hidden_state, attn_weights = self.self_attn(hidden_state, attention_mask=attention_mask)
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
        self.config = config
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

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
        cache: Optional[KVCache] = None
    ) -> mx.array:
        bsz, q_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        query_states = self.q_norm(query_states)

        if cross_attention_states is not None:
            key_states = self.k_proj(cross_attention_states)
            value_states = self.v_proj(cross_attention_states)
            key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            if cache is not None:
                # if we have a new image + new tokens, we only computed key_states on that new image
                # we still update the cross key states, past_image, new_image. And use it!
                keys, values = cache.update_and_fetch(keys, values)
        else:
            key_states, value_states = cache.state()

        key_states = self.k_norm(key_states)
        attn_output = mx.fast.scaled_dot_product_attention(query_states, key_states, value_states)

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
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.rope = initialize_rope(config)

    def __call__(
        self,
        hidden_states: mx.array,
        cache: Optional[KVCache] = None
    ):
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            query_states = self.rope(query_states, offset=cache.offset)
            key_states = self.rope(key_states, offset=cache.offset)
            key_states, value_states = cache.update_and_fetch(key_states, value_states)
        else:
            query_states = self.rope(query_states)
            key_states = self.rope(key_states)

        attn_output = mx.fast.scaled_dot_product_attention(query_states, key_states, value_states).transpose(0, 2, 1, 3).reshape(bsz, q_len, -1)
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
        cache: Optional[KVCache] = None,
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
        cache: Optional[KVCache] = None,
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
        self.scale = config.hidden_size**-0.5

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.class_embedding = self.scale * mx.random.uniform(shape=(self.hidden_size,))
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

        pixel_values = pixel_values.reshape(batch_size * num_concurrent_media * num_tiles, num_channels, height, width)
        aspect_ratio_ids = aspect_ratio_ids.reshape(batch_size * num_concurrent_media, -1)

        # Patch embedding
        patch_embeds = self.patch_embedding(pixel_values)
        hidden_state = patch_embeds.flatten(2).transpose(0, 2, 1, 3)

        # Tile embeddings
        _, num_patches, dim = hidden_state.shape
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media, num_tiles, -1, dim)
        hidden_state = self.pre_tile_positional_embedding(hidden_state, aspect_ratio_ids)

        # Add cls token
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media * num_tiles, num_patches, dim)
        batch_size, _, hidden_size = hidden_state.shape
        class_embedding = self.class_embedding.expand(batch_size, 1, hidden_size)
        hidden_state = mx.concatenate([class_embedding, hidden_state], axis=1)
        num_patches += 1

        # Position embeddings
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media, num_tiles, num_patches, dim)
        hidden_state = self.gated_positional_embedding(hidden_state, aspect_ratio_ids)
        hidden_state = self.layernorm_pre(hidden_state)

        # Compute the number of tokens to pad
        num_padding_patches = (8 - (hidden_state.shape[-2] % 8)) % 8
        # Compute padding tuple for pad function
        padding = (0, 0, 0, num_padding_patches)  # (pad_left, pad_right, pad_left for dim -2, pad_right for dim -2)
        # Pad the tensor
        hidden_state = mx.pad(hidden_state, padding, mode="constant", constant_values=0)
        slice_index = -num_padding_patches if num_padding_patches > 0 else None

        # Prepare attention mask
        attention_mask = aspect_ratio_mask.reshape(batch_size * num_concurrent_media, -1)
        attention_mask = _prepare_aspect_ratio_attention_mask(
            aspect_ratio_mask=attention_mask,
            num_patches=self.num_patches,
            target_length=hidden_state.shape[2]
        )

        # Apply encoder
        hidden_state = hidden_state.view(batch_size * num_concurrent_media, -1, dim)
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


class MllamaTextModel(nn.Module):
    def __init__(self, config: MllamaTextConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size + 8, config.hidden_size)
        self.cross_attention_layers = config.cross_attention_layers

        self.layers = []
        for layer_idx in range(config.num_hidden_layers):
            if layer_idx in self.cross_attention_layers:
                self.layers.append(MllamaCrossAttentionDecoderLayer(config, layer_idx))
            else:
                self.layers.append(MllamaSelfAttentionDecoderLayer(config, layer_idx))

        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        cross_attention_states: Optional[mx.array] = None,
        cache: Optional[List[KVCache]] = None,
        inputs_embeds: Optional[mx.array] = None
    ) -> mx.array:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        
        for idx, decoder_layer, c in enumerate(zip(self.layers, cache)):
            # For text-only path we should skip cross attention layers.
            # Let's check if the layer is cross attention layer and if we have cross attention states
            # or cached cross attention states.
            if (idx in self.cross_attention_layers) and cross_attention_states is None and not any(c.state()):
                continue
            
            hidden_states = decoder_layer(
                hidden_states,
                cross_attention_states=cross_attention_states,
                cache=c
            )
        return self.norm(hidden_states)
    
    
class MllamaModel(nn.Module):
    def __init__(self, config: MllamaConfig):
        self.vocab_size = config.text_config.vocab_size
        self.hidden_size = config.text_config.hidden_size
        self.max_num_tiles = config.vision_config.max_num_tiles
        self.vision_output_dim = config.vision_config.vision_output_dim
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

        self.vision_model = MllamaVisionModel(config.vision_config)
        self.language_model = MllamaTextModel(config.text_config)
        self.multi_modal_projector = nn.Linear(
            config.vision_config.vision_output_dim,
            config.text_config.hidden_size,
            bias=True,
        )

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        aspect_ratio_mask: Optional[mx.array] = None,
        aspect_ratio_ids: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
        inputs_embeds: Optional[mx.array] = None,
    ) -> mx.array:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if pixel_values is not None and cross_attention_states is not None:
            raise ValueError("`pixel_values` and `cross_attention_states` cannot be provided simultaneously")

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
            input_ids=input_ids,
            cross_attention_states=cross_attention_states,
            cache=cache
        )
        return outputs