"""Model-specific kernel fusion patches for MLX inference.

Detects model type after loading and applies optimized kernel patches.
Currently supports:
- Qwen3.5 MoE (model_type: qwen3_5_moe): batched fused oproj (GDN + GQA + MoE)

Set EXO_FUSED_KERNELS=0 to disable all patches (vanilla mode).
Default: EXO_FUSED_KERNELS=1 (enabled).
"""

import json
import os
from pathlib import Path

import mlx.nn as nn
from loguru import logger


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
        from .qwen3_5_moe.apply import apply_qwen35_batched_fused_patches

        logger.info("Detected Qwen3.5 MoE model, applying batched fused kernel patches")
        apply_qwen35_batched_fused_patches(model)

    elif model_type == "qwen3_5":
        from .qwen3_5.lpb_patch import apply_lpb_patches

        logger.info("Detected Qwen3.5 dense model, applying LpB kernel patches")
        apply_lpb_patches(model, batch_size=4)
