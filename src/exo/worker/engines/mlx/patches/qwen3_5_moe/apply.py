"""Apply batched fused kernel patches to Qwen3.5 MoE models.

Entry point called from patches/__init__.py after model type detection.

Installs the `fused_attn_batched_moe` mode (the TP-friendly default):
  - Fused GDN/GQA attention with out_proj/o_proj kept INSIDE the attention
    module (no oproj cross-boundary fusion). Preserves the natural TP
    all-reduce boundary that exo's QwenShardingStrategy relies on.
  - Batched merged SwiGLU + down_proj MoE (5 dispatches) with the new
    heterogeneous-grid kernel that runs the shared expert SwiGLU in parallel
    with softmax+topk routing.

For each layer the orchestrator:
  1. Stacks gate+up weights for the routed experts (_patch_swiglu_weights).
  2. Stacks gate+up weights and stores down weights for the shared expert
     (_patch_shared_expert).
  3. Stores down_proj weights for the routed experts (_patch_down_proj).
  4. Stores shared_expert_gate weights for the seg phase (_patch_seg_weights).
  5. Pre-merges attention projection weights (_patch_gdn_proj_weights or
     _patch_gqa_proj_weights).

Then class-level method replacement installs the new __call__ implementations.
The TP boundary is intact: out_proj/o_proj remain self-contained nn.Linear
calls and the MoE __call__ returns a (B, S, K) hidden state, so exo's
sharded_to_all_linear and ShardedMoE wrapper insert all_sum unchanged.

The legacy batched_fused_oproj mode functions (_batched_oproj_moe_call,
_fused_gdn_decoder_call, _patch_oproj_gate_rms) remain in the module for
reference and are not installed.
"""

import time

import mlx.core as mx
import mlx.nn as nn
from loguru import logger

from .common import (
    _patch_swiglu_weights,
    _patch_shared_expert,
    _patch_down_proj,
    _patch_seg_weights,
    _patch_gdn_proj_weights,
    _patch_gqa_proj_weights,
)
from mlx_lm.models.qwen3_5 import DecoderLayer, GatedDeltaNet
from mlx_lm.models.qwen3_next import Qwen3NextAttention, Qwen3NextSparseMoeBlock


def apply_qwen35_batched_fused_patches(model: nn.Module) -> None:
    """Install the fused_attn_batched_moe mode on every MoE decoder layer.

    Works with BatchGenerator for any batch size 1..8 at decode (S=1).
    Falls back to vanilla inside each patched __call__ for B>8 or S>1.
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
            _patch_seg_weights(moe)

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
    from .fused_gdn_attention import _fused_gdn_with_outproj_call
    from .batched_fused_gqa_attention import _batched_fused_gqa_with_oproj_call
    from .batched_moe import _batched_swiglu_down_moe_call_with_epilogue
    from .decoder import _fused_decoder_call

    # Class-level method replacement
    GatedDeltaNet.__call__ = _fused_gdn_with_outproj_call
    Qwen3NextAttention.__call__ = _batched_fused_gqa_with_oproj_call
    Qwen3NextSparseMoeBlock.__call__ = _batched_swiglu_down_moe_call_with_epilogue
    # _fused_decoder_call: h = x + attn(norm(x));  out = mlp(norm(h), _residual=h)
    # Attention wrappers return post-out_proj. MoE returns full output with
    # residual baked in by batched_moe_epilogue. Decoder body has no extra add.
    DecoderLayer.__call__ = _fused_decoder_call

    t_patch = time.time() - t0
    logger.info(
        f"Qwen3.5 batched fused: {n_gdn} GDN + {n_gqa} GQA layers, "
        f"{n_layers} total in {t_patch:.1f}s"
    )
