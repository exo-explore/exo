"""Apply oproj fusion patches to Qwen3.5 MoE models.

Entry point called from patches/__init__.py after model type detection.
"""

import time

import mlx.nn as nn
from loguru import logger

from .common import apply_oproj_fused_patches


def apply_qwen35_oproj_patches(model: nn.Module) -> None:
    """Apply oproj 4-dispatch fusion to all layers of a Qwen3.5 MoE model.

    Patches decoder, attention, and MoE __call__ methods to use custom Metal
    kernels for decode (seq_len=1). Prefill (seq_len>1) falls back to vanilla.
    """
    layers = model.layers  # type: ignore[attr-defined]
    n_layers = len(layers)

    t0 = time.time()
    apply_oproj_fused_patches(layers, gate_bm=8, free_originals=False)
    t_patch = time.time() - t0

    logger.info(f"Qwen3.5 oproj fusion: patched {n_layers} layers in {t_patch:.1f}s")
