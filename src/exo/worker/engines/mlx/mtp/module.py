"""MTP Module for DeepSeek V3 Multi-Token Prediction.

The MTP architecture predicts one additional token ahead using:
1. hnorm - RMSNorm for hidden state normalization
2. enorm - RMSNorm for embedding normalization
3. eh_proj - Linear(2*hidden_size -> hidden_size) projection
4. transformer_block - Single decoder layer (attention + MLP)
5. Shared embedding/lm_head from main model

Forward pass:
    h_norm = hnorm(hidden_state)
    e_norm = enorm(embed(token))
    projected = eh_proj(concat([h_norm, e_norm]))
    new_hidden = transformer_block(projected)
    logits = lm_head(output_norm(new_hidden))
"""

from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache
from mlx_lm.models.deepseek_v3 import (
    DeepseekV3Attention,
    DeepseekV3MLP,
    ModelArgs,
)

MTP_LAYER_INDEX = 61


class MTPModule(nn.Module):
    """Multi-Token Prediction module for DeepSeek V3.

    This module is initialized from the layer 61 weights that are normally
    discarded during model loading. It enables speculative decoding by
    predicting one token ahead using the hidden state from the main model.
    """

    def __init__(
        self,
        config: ModelArgs,
        shared_embedding: nn.Embedding,
        shared_lm_head: nn.Linear,
        output_norm: nn.RMSNorm,
    ) -> None:
        super().__init__()
        self.config = config

        # MTP-specific normalization layers
        self.hnorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.enorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Projection: concatenated [hidden, embedding] -> hidden_size
        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)

        # Single transformer block for MTP
        # Use a dense MLP since this is just a single layer
        self.transformer_block = MTPTransformerBlock(config)

        # Share embedding and lm_head with main model
        self._shared_embedding = shared_embedding
        self._shared_lm_head = shared_lm_head
        self._output_norm = output_norm

    def __call__(
        self,
        hidden_state: mx.array,
        draft_token: mx.array,
        cache: KVCache | None = None,
        mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Forward pass for MTP.

        Args:
            hidden_state: Hidden state from main model [batch, seq_len, hidden_size]
            draft_token: Token to embed and combine with hidden state [batch, seq_len]
            cache: Optional KV cache for the MTP transformer block
            mask: Optional attention mask

        Returns:
            tuple of (logits, new_hidden_state)
        """
        # Get embedding of draft token
        embedding = self._shared_embedding(draft_token)

        # Normalize hidden state and embedding
        h_norm = self.hnorm(hidden_state)
        e_norm = self.enorm(embedding)

        # Project concatenated representation
        concatenated = mx.concatenate([h_norm, e_norm], axis=-1)
        projected = self.eh_proj(concatenated)

        # Pass through single transformer block
        new_hidden = self.transformer_block(projected, mask=mask, cache=cache)

        # Apply output norm and get logits
        normed_hidden = self._output_norm(new_hidden)
        logits = self._shared_lm_head(normed_hidden)

        return logits, new_hidden


class MTPTransformerBlock(nn.Module):
    """Single transformer block for MTP.

    This is similar to DeepseekV3DecoderLayer but uses a dense MLP
    instead of MoE since this is just for the single MTP layer.
    """

    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.self_attn = DeepseekV3Attention(config)
        # MTP uses dense MLP, not MoE
        self.mlp = DeepseekV3MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        """Forward pass with residual connections."""
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


def extract_mtp_weights(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Extract MTP-specific weights from layer 61.

    The MTP layer has these weight patterns:
    - model.layers.61.enorm.weight -> MTP embedding normalization
    - model.layers.61.hnorm.weight -> MTP hidden normalization
    - model.layers.61.eh_proj.weight -> MTP projection layer
    - model.layers.61.self_attn.* -> MTP attention
    - model.layers.61.input_layernorm.* -> MTP layer norms
    - model.layers.61.post_attention_layernorm.*
    - model.layers.61.mlp.* -> MTP MLP (dense, not MoE)

    Args:
        weights: Full model weights dict

    Returns:
        Dict of MTP-specific weights with keys renamed for MTPModule
    """
    mtp_weights: dict[str, mx.array] = {}
    mtp_prefix = f"model.layers.{MTP_LAYER_INDEX}."

    for key, value in weights.items():
        if key.startswith(mtp_prefix):
            # Remove the layer prefix to get relative path
            new_key = key[len(mtp_prefix) :]
            mtp_weights[new_key] = value

    return mtp_weights


def load_mtp_weights_into_module(
    mtp_module: MTPModule,
    mtp_weights: dict[str, mx.array],
) -> None:
    """Load extracted MTP weights into the MTPModule.

    Args:
        mtp_module: The MTPModule instance to load weights into
        mtp_weights: Extracted MTP weights from extract_mtp_weights()
    """
    # Map weight names to module attributes
    weight_mapping: dict[str, str] = {
        "enorm.weight": "enorm.weight",
        "hnorm.weight": "hnorm.weight",
        "eh_proj.weight": "eh_proj.weight",
    }

    # Load direct mappings
    for src_name, dst_name in weight_mapping.items():
        if src_name in mtp_weights:
            parts = dst_name.split(".")
            obj: Any = mtp_module
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], mtp_weights[src_name])

    # Load transformer block weights (self_attn, mlp, layer norms)
    transformer_prefixes = [
        "self_attn",
        "mlp",
        "input_layernorm",
        "post_attention_layernorm",
    ]

    for prefix in transformer_prefixes:
        for key, value in mtp_weights.items():
            if key.startswith(prefix):
                # Navigate to the correct attribute
                parts = key.split(".")
                obj = mtp_module.transformer_block
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
