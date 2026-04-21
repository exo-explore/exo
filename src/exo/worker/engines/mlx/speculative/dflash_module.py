#!/usr/bin/env python3
"""DFlash draft model for Qwen3.5 on MLX.

Port of z-lab's DFlash (Block Diffusion for Flash Speculative Decoding).
A 5-layer bidirectional transformer that generates all draft tokens in one
parallel forward pass, conditioned on target model hidden states.

Reference: z-lab/dflash benchmark.py
"""

import json
import os
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache
from mlx_lm.models.activations import swiglu


class DFlashAttention(nn.Module):
    """Bidirectional attention with target context injection.

    Q from draft tokens. K/V from cat(target_context, draft_tokens).
    Non-causal: every draft position attends to every other + context.
    """

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, rope_theta,
                 rms_norm_eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=rms_norm_eps)
        self.rope = nn.RoPE(head_dim, base=rope_theta)

    def __call__(self, hidden_states, target_hidden, q_offset, cache=None):
        B, S, _ = hidden_states.shape

        # Q from draft only
        q = self.q_norm(
            self.q_proj(hidden_states).reshape(B, S, self.num_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)

        # K/V from cat(context, draft) — single projection each
        kv_input = mx.concatenate([target_hidden, hidden_states], axis=1)
        kv_len = kv_input.shape[1]
        k = self.k_norm(
            self.k_proj(kv_input).reshape(B, kv_len, self.num_kv_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        v = self.v_proj(kv_input).reshape(
            B, kv_len, self.num_kv_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        # RoPE — fused mx.fast.rope, 1 dispatch each
        q = self.rope(q, offset=q_offset)
        k = self.rope(k, offset=cache.offset)

        # KV cache — pre-allocated buffer, no concat copies
        keys, values = cache.update_and_fetch(k, v)

        # Bidirectional attention (no causal mask)
        output = mx.fast.scaled_dot_product_attention(
            q, keys, values, scale=self.scale)
        output = output.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.o_proj(output)


class DFlashDecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim,
                 intermediate_size, rope_theta, rms_norm_eps=1e-6):
        super().__init__()
        self.self_attn = DFlashAttention(
            hidden_size, num_heads, num_kv_heads, head_dim, rope_theta,
            rms_norm_eps=rms_norm_eps)
        self.mlp_gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.mlp_up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.mlp_down = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)

    def __call__(self, hidden_states, target_hidden, q_offset, cache=None):
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        h = self.self_attn(h, target_hidden, q_offset, cache)
        h = residual + h

        residual = h
        h = self.post_attention_layernorm(h)
        h = residual + self.mlp_down(swiglu(self.mlp_gate(h), self.mlp_up(h)))
        return h


class DFlashDrafter:
    """DFlash draft model for speculative decoding.

    Maintains a KV cache for the draft model across iterations.
    Each iteration:
      1. Receives target_hidden for accepted positions
      2. Processes [accepted_token, MASK, ...] with bidirectional attention
      3. Draft model catches up on accepted positions via KV cache
    """

    def __init__(self, target_model, dflash_model_path):
        self.target = target_model
        self._inner = getattr(target_model, 'model', None) or target_model.language_model.model
        self._text_model = getattr(target_model, 'model', None) or target_model.language_model

        # Load config
        if os.path.isdir(dflash_model_path):
            model_dir = dflash_model_path
        else:
            from huggingface_hub import snapshot_download
            model_dir = snapshot_download(dflash_model_path)

        with open(os.path.join(model_dir, 'config.json')) as f:
            config = json.load(f)

        self.hidden_size = config['hidden_size']
        self.num_heads = config['num_attention_heads']
        self.num_kv_heads = config['num_key_value_heads']
        self.head_dim = config.get('head_dim', self.hidden_size // self.num_heads)
        self.intermediate_size = config['intermediate_size']
        self.num_layers = config['num_hidden_layers']
        self.block_size = config['block_size']
        self.target_layer_ids = config['dflash_config']['target_layer_ids']
        self.mask_token_id = config['dflash_config']['mask_token_id']
        self.rope_theta = config.get('rope_theta', 10000000)
        self.rms_norm_eps = config.get('rms_norm_eps', 1e-6)

        print(f"  DFlash config: {self.num_layers} layers, hidden={self.hidden_size}, "
              f"heads={self.num_heads}/{self.num_kv_heads}, head_dim={self.head_dim}, "
              f"block={self.block_size}, rms_norm_eps={self.rms_norm_eps}")
        print(f"  Target layers: {self.target_layer_ids}, mask_token={self.mask_token_id}")

        # Build layers
        self.layers = []
        for i in range(self.num_layers):
            self.layers.append(DFlashDecoderLayer(
                self.hidden_size, self.num_heads, self.num_kv_heads,
                self.head_dim, self.intermediate_size, self.rope_theta,
                rms_norm_eps=self.rms_norm_eps))

        n_target = len(self.target_layer_ids)
        self.fc = nn.Linear(n_target * self.hidden_size, self.hidden_size, bias=False)
        self.hidden_norm = nn.RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.norm = nn.RMSNorm(self.hidden_size, eps=self.rms_norm_eps)

        # Shared from target
        self.embed_tokens = self._inner.embed_tokens
        if hasattr(self._text_model, 'lm_head'):
            self.lm_head = self._text_model.lm_head
        else:
            self.lm_head = None

        # Draft KV cache: KVCache per layer (pre-allocated buffers)
        self.draft_cache = None

        # Load weights
        weights = mx.load(os.path.join(model_dir, 'model.safetensors'))
        self._load_weights(weights)
        total_params = sum(w.size for w in weights.values())
        print(f"  DFlash loaded: {len(weights)} tensors, {total_params / 1e6:.1f}M params")

    def _load_weights(self, weights):
        self.fc.weight = weights['fc.weight']
        self.hidden_norm.weight = weights['hidden_norm.weight']
        self.norm.weight = weights['norm.weight']

        for i, layer in enumerate(self.layers):
            p = f'layers.{i}'
            attn = layer.self_attn
            attn.q_proj.weight = weights[f'{p}.self_attn.q_proj.weight']
            attn.k_proj.weight = weights[f'{p}.self_attn.k_proj.weight']
            attn.v_proj.weight = weights[f'{p}.self_attn.v_proj.weight']
            attn.o_proj.weight = weights[f'{p}.self_attn.o_proj.weight']
            attn.q_norm.weight = weights[f'{p}.self_attn.q_norm.weight']
            attn.k_norm.weight = weights[f'{p}.self_attn.k_norm.weight']
            layer.mlp_gate.weight = weights[f'{p}.mlp.gate_proj.weight']
            layer.mlp_up.weight = weights[f'{p}.mlp.up_proj.weight']
            layer.mlp_down.weight = weights[f'{p}.mlp.down_proj.weight']
            layer.input_layernorm.weight = weights[f'{p}.input_layernorm.weight']
            layer.post_attention_layernorm.weight = weights[f'{p}.post_attention_layernorm.weight']

    def reset_draft_cache(self):
        self.draft_cache = [KVCache() for _ in range(self.num_layers)]

    def crop_draft_cache(self, length):
        """Crop draft KV cache to given length (discard speculative entries)."""
        for cache in self.draft_cache:
            cache.offset = length

    def get_target_hidden(self, inputs, cache):
        """Run target model, capture hidden states at specified layers + logits."""
        model = self._inner
        if hasattr(model, 'embed_tokens'):
            hidden_states = model.embed_tokens(inputs)
        else:
            hidden_states = inputs

        cache_list = cache if cache is not None else [None] * len(model.layers)

        from mlx_lm.models.qwen3_5 import create_attention_mask, create_ssm_mask
        fa_mask = create_attention_mask(hidden_states, cache_list[model.fa_idx])
        ssm_mask = create_ssm_mask(hidden_states, cache_list[model.ssm_idx])

        layer_hiddens = {}
        for i, (layer, c) in enumerate(zip(model.layers, cache_list)):
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden_states = layer(hidden_states, mask=mask, cache=c)
            if i in self.target_layer_ids:
                layer_hiddens[i] = hidden_states

        selected = [layer_hiddens[i] for i in self.target_layer_ids]
        target_hidden = mx.concatenate(selected, axis=-1)

        normed = model.norm(hidden_states)
        if self.lm_head is not None:
            logits = self.lm_head(normed)
        else:
            logits = self.embed_tokens.as_linear(normed)

        return target_hidden, logits

    def draft(self, target_hidden, block_output_ids, start):
        """Run DFlash draft model.

        Args:
            target_hidden: (B, accepted_len, n_layers * hidden) — hidden states for
                           accepted positions from the previous verify step
            block_output_ids: (B, block_size) — [accepted_token, MASK, ..., MASK]
            start: int — position of the first token in the block
        Returns:
            draft_logits: (B, block_size-1, vocab) — logits for mask positions
        """
        bs = self.block_size

        # Noise embedding from target's embed_tokens
        noise_embedding = self.embed_tokens(block_output_ids)  # (B, bs, hidden)

        # Project target hidden states
        ctx = self.hidden_norm(self.fc(target_hidden))  # (B, accepted_len, hidden)

        # Run through layers
        h = noise_embedding
        for i, layer in enumerate(self.layers):
            h = layer(h, ctx, q_offset=start, cache=self.draft_cache[i])

        # Final norm + lm_head on mask positions (1..bs-1)
        h = self.norm(h)
        draft_hidden = h[:, -(bs - 1):, :]

        if self.lm_head is not None:
            draft_logits = self.lm_head(draft_hidden)
        else:
            draft_logits = self.embed_tokens.as_linear(draft_hidden)

        return draft_logits
