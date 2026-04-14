from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map_with_path
from mlx_lm.convert import (
    MODEL_CONVERSION_DTYPES as _MODEL_CONVERSION_DTYPES_RAW,  # pyright: ignore[reportAny]
)
from mlx_lm.utils import load_model as _mlx_lm_load_model

MODEL_CONVERSION_DTYPES: list[str] = cast(list[str], _MODEL_CONVERSION_DTYPES_RAW)


def _patched_convert(
    hf_path: Path,
    lazy: bool = True,
    strict: bool = False,
    dtype: str | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    model, config = _mlx_lm_load_model(hf_path, lazy=lazy, strict=strict)

    if dtype is None:
        raw = config.get("torch_dtype")
        dtype = raw if isinstance(raw, str) else None
    if dtype is None and isinstance(text_config := config.get("text_config"), dict):
        raw = cast(dict[str, Any], text_config).get("dtype")
        dtype = raw if isinstance(raw, str) else None
    target_dtype: mx.Dtype | None = (
        cast(mx.Dtype, getattr(mx, dtype))
        if dtype is not None and dtype in MODEL_CONVERSION_DTYPES
        else None
    )

    def default_predicate(_: str) -> bool:
        return True

    cast_predicate = cast(
        Callable[[str], bool], getattr(model, "cast_predicate", default_predicate)
    )

    def normalize(path: str, value: mx.array) -> mx.array:
        if not mx.issubdtype(value.dtype, mx.floating):
            return value
        if target_dtype is not None and cast_predicate(path):
            value = value.astype(target_dtype)
        return mx.contiguous(value)

    params = cast(dict[str, Any], model.parameters())
    model.update(cast(dict[str, Any], tree_map_with_path(normalize, params)))

    return model, config


def load_model(
    model_path: Path,
    lazy: bool = False,
    strict: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    return _patched_convert(model_path, lazy=lazy, strict=strict)
