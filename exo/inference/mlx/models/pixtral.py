import inspect
from dataclasses import dataclass
from typing import Optional, List

import mlx.core as mx
import mlx.nn as nn


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
        mesh = mx.meshgrid(mx.arange(height), mx.arange(width), indexing="ij")
        h_grid, v_grid = mx.stack(mesh, dim=-1).reshape(-1, 2).chunk(2, -1)
        ids = h_grid * max_width + v_grid
        positions.append(ids[:, 0])
        mx.meshgrid
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
        self.inv_freq = mx.concatenate(
            [
                freqs_h[:, None, :].repeat(1, max_patches_per_side, 1),
                freqs_w[None, :, :].repeat(max_patches_per_side, 1, 1),
            ],
            dim=-1,
        ).reshape(-1, dim // 2)

    def __call__(self, x, position_ids):
        freqs = self.inv_freq[position_ids]
        return mx.cos(freqs), mx.sin(freqs)


def rotate_half(x: mx.array) -> mx.array:
    x1, x2 = mx.split(x, 2, axis=-1)
    return mx.concatenate([-x2, x1], axis=-1)

def apply_rotary_pos_emb(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array, unsqueeze_dim: int = 1) -> tuple[mx.array, mx.array]:
    cos = mx.expand_dims(cos, axis=unsqueeze_dim)
    sin = mx.expand_dims(sin, axis=unsqueeze_dim)
    
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
        
        queries = queries.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin, unsqueeze_dim=0)

        scores = (queries * self.scale) @ keys

        if attention_mask is not None:
            scores = scores + attention_mask.astype(scores.dtype)

        scores = mx.softmax(scores, axis=-1)
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(values_hat)


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
        patch_embeds_list = [self.patch_conv(mx.expand_dims(img, axis=0)) for img in pixel_values]

        patch_embeds = mx.concatenate([p.reshape(p.shape[0], p.shape[1], -1).transpose(0, 2, 1) for p in patch_embeds_list], axis=1)
        patch_embeds = self.ln_pre(patch_embeds)

        position_ids = position_ids_in_meshgrid(
            patch_embeds_list, max_width=self.config.image_size // self.config.patch_size
        )

        position_embedding = self.patch_positional_embedding(patch_embeds, position_ids)
        attention_mask = generate_block_attention_mask(
            [p.shape[-2] * p.shape[-1] for p in patch_embeds_list], patch_embeds
        )
        return self.transformer(patch_embeds, attention_mask, position_embedding)
