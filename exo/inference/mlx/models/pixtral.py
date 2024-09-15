import math
import inspect
from dataclasses import dataclass, field
from typing import Optional, Dict, Union, List

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import BaseModelArgs, KVCache
from exo.inference.shard import Shard
from .base import IdentityBlock
import numpy as np


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
        height, width = patch.shape[-2:]
        mesh = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        h_grid, v_grid = torch.stack(mesh, dim=-1).reshape(-1, 2).chunk(2, -1)
        ids = h_grid * max_width + v_grid
        positions.append(ids[:, 0])
    return torch.cat(positions)


class PixtralRotaryEmbedding(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.rope_type = "default"
        self.dim = config.head_dim
        self.base = config.rope_theta
        max_patches_per_side = config.image_size // config.patch_size
        freqs = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))

        h = torch.arange(max_patches_per_side, device=freqs.device)
        w = torch.arange(max_patches_per_side, device=freqs.device)

        freqs_h = torch.outer(h, freqs[::2]).float()
        freqs_w = torch.outer(w, freqs[1::2]).float()
        inv_freq = torch.cat(
            [
                freqs_h[:, None, :].repeat(1, max_patches_per_side, 1),
                freqs_w[None, :, :].repeat(max_patches_per_side, 1, 1),
            ],
            dim=-1,
        ).reshape(-1, self.dim // 2)  # we reshape to only index on the position indexes, not tuple of indexes
        # Different from paper, but it uses a different permutation in order to obtain the same calculation

        # TODO maybe make it torch compatible later on. We can also just slice
        self.register_buffer("inv_freq", torch.cat((inv_freq, inv_freq), dim=-1), persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        freqs = self.inv_freq[position_ids]
        # position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            emb = freqs
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class PixtralAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, patches, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=0)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, patches, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class PixtralMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.gelu = nn.GELU()

    def __call__(self, h):
        return self.down_proj(self.gelu(self.gate_proj(h)) * self.up_proj(h))


class PixtralRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps

    def forward(self, h: mx.array):
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
        self.config = config
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
    causal_mask = mx.full((seq_len, seq_len), fill_value=-float('inf'))

    block_end_idx = mx.cumsum(mx.array(patch_embeds_list), axis=-1)
    block_start_idx = mx.cumsum(mx.array([0] + patch_embeds_list[:-1]), axis=-1)
    
    for start, end in zip(block_start_idx, block_end_idx):
        causal_mask = causal_mask.at[start:end, start:end].set(0)

    causal_mask = mx.broadcast_to(causal_mask[None, None, :, :], (input_array.shape[0], 1, seq_len, seq_len))
    return causal_mask


class PixtralModel(nn.Module):
    def __init__(self, config: PixtralVisionConfig):
        super().__init__(config)
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
        # pass images through initial convolution independently
        patch_embeds_list = [self.patch_conv(mx.expand_dims(img, axis=0)) for img in pixel_values]

        # flatten to a single sequence
        patch_embeds = mx.concatenate([p.reshape(p.shape[0], p.shape[1], -1).transpose(0, 2, 1) for p in patch_embeds_list], axis=1)
        patch_embeds = self.ln_pre(patch_embeds)

        # positional embeddings
        position_ids = position_ids_in_meshgrid(
            patch_embeds_list, max_width=self.config.image_size // self.config.patch_size
        )

        position_embedding = self.patch_positional_embedding(patch_embeds, position_ids)
        attention_mask = generate_block_attention_mask(
            [p.shape[-2] * p.shape[-1] for p in patch_embeds_list], patch_embeds
        )
        return self.transformer(patch_embeds, attention_mask, position_embedding)
