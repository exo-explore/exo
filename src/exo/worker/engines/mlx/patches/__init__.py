"""Model-specific kernel fusion patches for MLX inference.

Detects model type after loading and applies optimized kernel patches
where available. Currently supports:
- Qwen3.5 MoE (model_type: qwen3_5_moe):
    EXO_FUSED_KERNELS=1: oproj mode (4-dispatch MoE fusion)
    EXO_FUSED_KERNELS=2: fused_gqa_gdn mode (full GDN + GQA + MoE fusion, default)

Set EXO_FUSED_KERNELS=0 to disable all patches (vanilla mode).
Default: EXO_FUSED_KERNELS=2 (fused_gqa_gdn, best performance).
"""

import json
import os
from pathlib import Path

import mlx.nn as nn
from loguru import logger


def maybe_apply_patches(model: nn.Module, model_path: Path) -> None:
    """Detect model type and apply kernel fusion patches if available."""
    fused_mode = os.environ.get("EXO_FUSED_KERNELS", "2")
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
        if fused_mode == "1":
            from .qwen3_5_moe.apply import apply_qwen35_oproj_patches

            logger.info("Detected Qwen3.5 MoE model, applying oproj fusion patches")
            apply_qwen35_oproj_patches(model)
        else:
            from .qwen3_5_moe.apply import apply_qwen35_fused_gqa_gdn_patches

            logger.info("Detected Qwen3.5 MoE model, applying fused GQA+GDN patches")
            apply_qwen35_fused_gqa_gdn_patches(model)
