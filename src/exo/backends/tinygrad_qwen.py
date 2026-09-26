"""Qwen2 and Qwen3 decoder blocks.

Qwen2 matches the Llama block, including optional attention bias. Qwen3
requires query and key normalization weights and otherwise uses that block.
"""

from __future__ import annotations

from tinygrad.tensor import Tensor

from exo.backends.tinygrad_llama import (
    DecoderLayerParameters,
    forward_llama_decoder_layer,
    require_qwen3_query_key_normalization,
)
from exo.backends.tinygrad_weights import TransformerArchitecture


def forward_qwen2_decoder_layer(
    hidden_state: Tensor,
    layer: DecoderLayerParameters,
    architecture: TransformerArchitecture,
    cached_keys: Tensor | None,
    cached_values: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run one Qwen2 block. The math matches Llama."""
    return forward_llama_decoder_layer(
        hidden_state, layer, architecture, cached_keys, cached_values
    )


def forward_qwen3_decoder_layer(
    hidden_state: Tensor,
    layer: DecoderLayerParameters,
    architecture: TransformerArchitecture,
    cached_keys: Tensor | None,
    cached_values: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run one Qwen3 block after requiring query and key normalization.

    Raises:
        TinygradModelSupportError: The runner entrypoint handles this when
            the Qwen3 layer has no ``q_norm`` or ``k_norm`` weight.
    """
    require_qwen3_query_key_normalization(layer)
    return forward_llama_decoder_layer(
        hidden_state, layer, architecture, cached_keys, cached_values
    )
