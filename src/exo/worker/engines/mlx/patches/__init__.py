import json
import os
from pathlib import Path

import mlx.nn as nn
from loguru import logger

from exo.worker.engines.mlx.patches.opt_batch_gen import apply_batch_gen_patch
from exo.worker.engines.mlx.patches.standard_yarn_rope import patch_yarn_rope

_applied = False


def apply_mlx_patches() -> None:
    global _applied
    if _applied:
        return
    _applied = True
    patch_yarn_rope()
    # Skip fast_next patch when speculative is enabled — it bypasses _next()
    # which MTPBatchGenerator overrides for speculative decoding
    if os.environ.get("EXO_SPECULATIVE") != "1":
        apply_batch_gen_patch()


def _qwen35_moe_patches_supported(model: nn.Module) -> tuple[bool, str]:
    """Check whether the qwen3_5_moe batched fused patches can run on this model.

    The patches' fused kernels and `_fused_shared_expert` hardcode 8-bit
    quantized_matmul calls (see `qwen3_5_moe/batched_moe.py`). They were
    validated against models whose shared expert is 8-bit. Applying them to
    a model with 4-bit (or other) shared-expert weights crashes at warmup
    inside `mx.quantized_matmul` with a shape/bits mismatch.

    Probe the loaded model directly (config.json doesn't always reflect
    actual on-disk quantization for mixed-precision quants) and return
    False with a reason if the assumption is broken.
    """
    try:
        layers = model.language_model.model.layers  # type: ignore[attr-defined]
    except AttributeError:
        return False, "model has no language_model.model.layers attribute"

    for layer in layers:
        moe = getattr(layer, "mlp", None)
        if moe is None or not hasattr(moe, "shared_expert"):
            continue
        sgp = getattr(moe.shared_expert, "gate_proj", None)
        bits = getattr(sgp, "bits", None)
        if bits is None:
            return False, "shared expert is not quantized"
        if bits != 8:
            return False, f"shared expert is {bits}-bit (patches require 8-bit)"
        return True, ""

    return False, "no MoE layers found in model"


def maybe_apply_patches(model: nn.Module, model_path: Path) -> None:
    """Detect model type and apply kernel fusion patches if available."""
    fused_mode = os.environ.get("EXO_FUSED_KERNELS", "1")
    if fused_mode == "0":
        logger.info("Kernel fusion patches disabled (EXO_FUSED_KERNELS=0)")
        return

    config_path = model_path / "config.json"
    if not config_path.exists():
        return

    with open(config_path) as f:
        config = json.load(f)

    model_type = config.get("model_type", "")

    if model_type == "qwen3_5_moe":
        supported, reason = _qwen35_moe_patches_supported(model)
        if not supported:
            logger.warning(
                f"Skipping Qwen3.5 MoE batched fused patches: {reason}. "
                f"Model will run via the vanilla MLX path."
            )
            return

        from .qwen3_5_moe.apply import apply_qwen35_batched_fused_patches

        logger.info("Detected Qwen3.5 MoE model, applying batched fused kernel patches")
        apply_qwen35_batched_fused_patches(model)

    elif model_type == "qwen3_5":
        from .qwen3_5.lpb_patch import apply_lpb_patches

        logger.info("Detected Qwen3.5 dense model, applying LpB kernel patches")
        apply_lpb_patches(model, batch_size=4)
