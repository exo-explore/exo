# Copyright © 2026 Apple Inc.

"""Inkling 2 (Inkling 2 Mini: config model_type inkling2_loop), for exo.

Vendored from mlx-lm commit 0c4c99192230f1943dd468d297f0b8a64ae7b7c2 (ml-explore/mlx-lm#2086, not merged),
mlx_lm/models/inkling2/, MIT licence, Copyright 2026 Apple Inc. Only the imports are changed: exo's mlx-lm pin
has the same helpers under their pre-#2082 paths (switch_layers, pipeline, base, kernels._common) and no
``splice`` (copied to splice.py). The processor (image preprocessing) is not vendored: exo serves it text only.

https://huggingface.co/thinkingmachines/Inkling-2-Small
"""

from exo.worker.engines.mlx.models.inkling.inkling2.config import ModelArgs
from exo.worker.engines.mlx.models.inkling.inkling2.model import Model

__all__ = [
    "Model",
    "ModelArgs",
]
