# type: ignore
# Vendored from ml-explore/mlx-lm @ 0c4c99192230f1943dd468d297f0b8a64ae7b7c2 (PR #2086, Inkling 2), mlx_lm/models/inkling2/vision.py
# (splice.py: mlx_lm/layers/common.py). MIT licence, Copyright 2026 Apple Inc. Local change: imports only (see __init__.py).
# Copyright © 2026 Apple Inc.

import mlx.core as mx
from mlx import nn

from exo.worker.engines.mlx.models.inkling.inkling2.config import VisionArgs

__all__ = [
    "VisionAttention",
    "VisionEmbeddings",
    "VisionEncoderLayer",
    "VisionMLP",
    "VisionMerger",
    "VisionModel",
    "VisionRotaryEmbedding",
]


def _precise_gelu(x: mx.array) -> mx.array:
    return nn.gelu(x.astype(mx.float32)).astype(x.dtype)


def _grid_positions(grid_h: int, grid_w: int, merge: int) -> tuple[mx.array, mx.array]:
    i = mx.arange(grid_h * grid_w)
    blocks_w = grid_w // merge
    row = i // (merge * merge * blocks_w) * merge + i // merge % merge
    col = i // (merge * merge) % blocks_w * merge + i % merge
    return row, col


def _bilinear_taps(index: mx.array, size: int, side: int) -> tuple[mx.array, mx.array]:
    src = (index.astype(mx.float32) + 0.5) * side / size - 0.5
    taps = mx.floor(src)[:, None] + mx.arange(2)
    weights = mx.maximum(1.0 - mx.abs(src[:, None] - taps), 0.0)
    return mx.clip(taps, 0, side - 1).astype(mx.int32), weights


class VisionEmbeddings(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        self.merge_size = args.adapter_merge_size
        self.grid_size = args.posemb_grid_size
        patch_dims = args.num_channels * args.native_patch_size**2
        self.patch_embedding = nn.Linear(patch_dims, args.hidden_size, bias=False)
        self.position_embedding = nn.Embedding(
            args.posemb_grid_size**2, args.hidden_size
        )
        self.class_embedding = mx.zeros((args.hidden_size,))
        self.cls_position_embedding = mx.zeros((args.hidden_size,))

    def __call__(self, pixel_values: mx.array, grid_h: int, grid_w: int) -> mx.array:
        patch_embeddings = self.patch_embedding(pixel_values)

        side = self.grid_size
        row, col = _grid_positions(grid_h, grid_w, self.merge_size)
        row_taps, row_weights = _bilinear_taps(row, grid_h, side)
        col_taps, col_weights = _bilinear_taps(col, grid_w, side)
        interp_indices = row_taps[:, :, None] * side + col_taps[:, None, :]
        interp_weights = row_weights[:, :, None] * col_weights[:, None, :]
        position_embeddings = self.position_embedding(interp_indices.reshape(-1, 4))
        position_embeddings = position_embeddings.astype(mx.float32)
        position_embeddings = position_embeddings * interp_weights.reshape(-1, 4, 1)
        patch_embeddings = patch_embeddings + position_embeddings.sum(axis=1).astype(
            patch_embeddings.dtype
        )

        cls_embedding = self.class_embedding + self.cls_position_embedding
        cls_embedding = cls_embedding.astype(patch_embeddings.dtype)
        return mx.concatenate([cls_embedding[None], patch_embeddings])


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        dims = args.head_dim // 2
        self._inv_freq = 1.0 / (
            args.rope_theta ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
        )

    def __call__(self, row: mx.array, col: mx.array) -> tuple[mx.array, mx.array]:
        freqs_h = row[:, None].astype(mx.float32) * self._inv_freq
        freqs_w = col[:, None].astype(mx.float32) * self._inv_freq
        freqs = mx.repeat(mx.concatenate([freqs_w, freqs_h], axis=-1), 2, axis=-1)
        return mx.cos(freqs), mx.sin(freqs)


def _apply_rotary_pos_emb_vision(
    hidden_states: mx.array, cos: mx.array, sin: mx.array
) -> mx.array:
    states = hidden_states.astype(mx.float32)
    pairs = states.reshape(*states.shape[:-1], -1, 2)
    rotated = mx.stack([-pairs[..., 1], pairs[..., 0]], axis=-1).reshape(states.shape)
    return (states * cos + rotated * sin).astype(hidden_states.dtype)


class VisionAttention(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        dims = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.scaling = args.head_dim**-0.5
        self.q_proj = nn.Linear(dims, dims)
        self.k_proj = nn.Linear(dims, dims)
        self.v_proj = nn.Linear(dims, dims)
        self.out_proj = nn.Linear(dims, dims)

    def __call__(
        self, hidden_states: mx.array, cos: mx.array, sin: mx.array
    ) -> mx.array:
        N = hidden_states.shape[0]
        shape = (N, self.num_heads, -1)
        cos, sin = cos[:, None], sin[:, None]
        query_states = self.q_proj(hidden_states).reshape(shape)
        key_states = self.k_proj(hidden_states).reshape(shape)
        value_states = self.v_proj(hidden_states).reshape(shape)
        query_states = _apply_rotary_pos_emb_vision(query_states, cos, sin)
        key_states = _apply_rotary_pos_emb_vision(key_states, cos, sin)
        query_states, key_states, value_states = (
            states.transpose(1, 0, 2)[None]
            for states in (query_states, key_states, value_states)
        )
        attn_output = mx.fast.scaled_dot_product_attention(
            query_states, key_states, value_states, scale=self.scaling
        )
        attn_output = attn_output[0].transpose(1, 0, 2).reshape(N, -1)
        return self.out_proj(attn_output)


class VisionMLP(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        self.fc1 = nn.Linear(args.hidden_size, args.intermediate_size)
        self.fc2 = nn.Linear(args.intermediate_size, args.hidden_size)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.fc2(_precise_gelu(self.fc1(hidden_states)))


class VisionEncoderLayer(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        dims, eps = args.hidden_size, args.layer_norm_eps
        self.self_attn = VisionAttention(args)
        self.mlp = VisionMLP(args)
        self.layernorm_before = nn.LayerNorm(dims, eps=eps)
        self.layernorm_after = nn.LayerNorm(dims, eps=eps)
        self.lambda_1 = mx.ones((dims,))
        self.lambda_2 = mx.ones((dims,))

    def __call__(
        self, hidden_states: mx.array, cos: mx.array, sin: mx.array
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.self_attn(self.layernorm_before(hidden_states), cos, sin)
        hidden_states = residual + self.lambda_1 * hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.layernorm_after(hidden_states))
        return residual + self.lambda_2 * hidden_states


class VisionMerger(nn.Module):
    def __init__(self, args: VisionArgs, text_hidden_size: int):
        super().__init__()
        dims = args.adapter_merge_size**2 * args.hidden_size
        hidden, bias = args.projection_hidden_dim, args.projector_bias
        self.linear_1 = nn.Linear(dims, hidden, bias=bias)
        self.linear_2 = nn.Linear(hidden, text_hidden_size, bias=bias)

    def __call__(self, image_features: mx.array) -> mx.array:
        return self.linear_2(_precise_gelu(self.linear_1(image_features)))


class VisionModel(nn.Module):
    def __init__(self, args: VisionArgs, text_hidden_size: int):
        super().__init__()
        self.args = args
        self.embeddings = VisionEmbeddings(args)
        self.pre_layernorm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)
        self.rotary_emb = VisionRotaryEmbedding(args)
        self.layers = [VisionEncoderLayer(args) for _ in range(args.num_hidden_layers)]
        self.merger = VisionMerger(args, text_hidden_size)

    def __call__(self, pixel_values: mx.array) -> mx.array:
        C, H, W = pixel_values.shape
        p, m = self.args.native_patch_size, self.args.adapter_merge_size
        if H % (p * m) or W % (p * m):
            raise ValueError(f"Image size {H}x{W} is not a multiple of {p * m}.")
        rows, cols = H // (p * m), W // (p * m)
        patches = pixel_values.transpose(1, 2, 0).reshape(rows, m, p, cols, m, p, C)
        # The merge blocks in raster order, then the patches of each block.
        patches = patches.transpose(0, 3, 1, 4, 2, 5, 6).reshape(-1, p * p * C)
        dtype = self.embeddings.patch_embedding.weight.dtype
        hidden_states = self.embeddings(patches.astype(dtype), rows * m, cols * m)
        hidden_states = self.pre_layernorm(hidden_states)

        row, col = _grid_positions(rows * m, cols * m, m)
        zero = mx.zeros((1,), dtype=row.dtype)
        cos, sin = self.rotary_emb(
            mx.concatenate([zero, row + 1]), mx.concatenate([zero, col + 1])
        )
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin)
        patch_states = hidden_states[1:].reshape(rows * cols, -1)
        return self.merger(patch_states)
