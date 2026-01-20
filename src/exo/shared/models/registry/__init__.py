"""Model registry with base models and quantization variants.

This module provides access to the model registry data stored in JSON files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final, cast

_REGISTRY_DIR: Final[Path] = Path(__file__).parent

# Load base models and variants data at import time
with (_REGISTRY_DIR / "base_models.json").open("r", encoding="utf-8") as f:
    _base_models_data: dict[str, Any] = cast(dict[str, Any], json.load(f))
    BASE_MODELS: Final[list[dict[str, object]]] = cast(
        list[dict[str, object]], _base_models_data["base_models"]
    )

with (_REGISTRY_DIR / "variants.json").open("r", encoding="utf-8") as f:
    _variants_data: dict[str, Any] = cast(dict[str, Any], json.load(f))
    VARIANTS: Final[list[dict[str, object]]] = cast(
        list[dict[str, object]], _variants_data["variants"]
    )

# Create lookup dicts for fast access
BASE_MODELS_BY_ID: Final[dict[str, dict[str, object]]] = {
    str(m["id"]): m for m in BASE_MODELS
}

# Variants are keyed by model_id (full HuggingFace repo path)
VARIANTS_BY_MODEL_ID: Final[dict[str, dict[str, object]]] = {
    str(v["model_id"]): v for v in VARIANTS
}

VARIANTS_BY_BASE_MODEL: Final[dict[str, list[dict[str, object]]]] = {}
for _variant in VARIANTS:
    _base_model_id = str(_variant["base_model"])
    if _base_model_id not in VARIANTS_BY_BASE_MODEL:
        VARIANTS_BY_BASE_MODEL[_base_model_id] = []
    VARIANTS_BY_BASE_MODEL[_base_model_id].append(_variant)

__all__ = [
    "BASE_MODELS",
    "VARIANTS",
    "BASE_MODELS_BY_ID",
    "VARIANTS_BY_MODEL_ID",
    "VARIANTS_BY_BASE_MODEL",
]
