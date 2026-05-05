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
    apply_batch_gen_patch()


def maybe_apply_patches(model: nn.Module, model_path: Path) -> None:
    """Detect model type and apply kernel fusion patches if available.

    Currently dispatches Qwen3.5 MoE models to the batched fused kernel mode
    (`fused_attn_batched_moe`). Set EXO_FUSED_KERNELS=0 to disable.
    """
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
