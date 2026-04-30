"""Apply batched fused kernel patches to Qwen3.5 MoE models.

Entry point called from patches/__init__.py after model type detection.
"""

import time

import mlx.core as mx
import mlx.nn as nn
from loguru import logger

from .common import (
    _patch_swiglu_weights,
    _patch_shared_expert,
    _patch_down_proj,
    _patch_oproj_gate_rms,
    _patch_gdn_proj_weights,
    _patch_gqa_proj_weights,
)
from mlx_lm.models.qwen3_5 import DecoderLayer
from mlx_lm.models.qwen3_next import Qwen3NextAttention, Qwen3NextSparseMoeBlock
from mlx_lm.models.qwen3_5 import GatedDeltaNet


def apply_qwen35_batched_fused_patches(model: nn.Module) -> None:
    """Apply batched fused patches (GDN + GQA attention + oproj MoE) to all layers.

    Fused GDN attention (3/4 layers) + fused GQA projections (1/4 layers)
    + batched oproj MoE (4 custom dispatches). Works with BatchGenerator for
    any batch size 1..8. Falls back to vanilla for B>8 or S>1.
    """
    layers = model.layers  # type: ignore[attr-defined]
    n_layers = len(layers)

    t0 = time.time()

    n_gdn = 0
    n_gqa = 0
    for li, layer in enumerate(layers):
        moe = layer.mlp
        if isinstance(moe, Qwen3NextSparseMoeBlock):
            # MoE weight prep
            _patch_swiglu_weights(moe)
            _patch_shared_expert(moe)
            _patch_down_proj(moe)
            _patch_oproj_gate_rms(layer, gate_bm=8)

            # Attention weight prep
            if layer.is_linear:
                _patch_gdn_proj_weights(layer.linear_attn)
                n_gdn += 1
            else:
                _patch_gqa_proj_weights(layer.self_attn)
                n_gqa += 1

            mx.clear_cache()  # release freed originals back to OS each layer

            if (li + 1) % 10 == 0 or li == 0:
                logger.info(f"  Patched layer {li+1}/{n_layers}")

    # Import patched __call__ methods
    from .fused_gdn_attention import _fused_gdn_call
    from .batched_fused_gqa_attention import _batched_fused_gqa_call
    from .batched_moe import _batched_oproj_moe_call
    from .decoder import _fused_gdn_decoder_call

    # Class-level method replacement
    GatedDeltaNet.__call__ = _fused_gdn_call
    Qwen3NextAttention.__call__ = _batched_fused_gqa_call
    Qwen3NextSparseMoeBlock.__call__ = _batched_oproj_moe_call
    DecoderLayer.__call__ = _fused_gdn_decoder_call

    t_patch = time.time() - t0
    logger.info(
        f"Qwen3.5 batched fused: {n_gdn} GDN + {n_gqa} GQA layers, "
        f"{n_layers} total in {t_patch:.1f}s"
    )
