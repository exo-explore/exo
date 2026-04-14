import glob
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map_with_path
from mlx_lm.convert import (
    MODEL_CONVERSION_DTYPES as _MODEL_CONVERSION_DTYPES_RAW,  # pyright: ignore[reportAny]
)
from mlx_lm.utils import (
    _get_classes as _raw_get_classes,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType]
)
from mlx_lm.utils import (
    load_config as _raw_load_config,  # pyright: ignore[reportUnknownVariableType]
)
from mlx_lm.utils import (
    load_model as _mlx_lm_load_model_raw,
)
from mlx_lm.utils import (
    quantize_model as _raw_quantize_model,  # pyright: ignore[reportUnknownVariableType]
)

MODEL_CONVERSION_DTYPES: list[str] = cast(list[str], _MODEL_CONVERSION_DTYPES_RAW)

_get_classes: Callable[..., tuple[Any, Any]] = cast(
    Callable[..., tuple[Any, Any]], _raw_get_classes
)
load_config: Callable[[Path], dict[str, Any]] = cast(
    Callable[[Path], dict[str, Any]], _raw_load_config
)
quantize_model: Callable[..., tuple[nn.Module, dict[str, Any]]] = cast(
    Callable[..., tuple[nn.Module, dict[str, Any]]], _raw_quantize_model
)
_mlx_lm_load_model: Callable[..., tuple[nn.Module, dict[str, Any]]] = cast(
    Callable[..., tuple[nn.Module, dict[str, Any]]], _mlx_lm_load_model_raw
)


def load_model(
    model_path: Path,
    lazy: bool = False,
    strict: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    return _patched_convert(model_path, lazy=lazy, strict=strict)


def _patched_convert(
    hf_path: Path,
    lazy: bool = True,
    strict: bool = False,
    dtype: str | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    config = load_config(hf_path)
    fp8_mode = _detect_fp8_mode(config)

    if fp8_mode is not None:
        model, config = _load_fp8_as_affine8(hf_path, config, fp8_mode, strict=strict)
    else:
        model, config = _mlx_lm_load_model(hf_path, lazy=True, strict=strict)

    _cast_and_contiguous(model, config, dtype=dtype)

    if not lazy:
        mx.eval(model.parameters())
    return model, config


def _cast_and_contiguous(
    model: nn.Module, config: dict[str, Any], dtype: str | None
) -> None:
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


def _detect_fp8_mode(config: dict[str, Any]) -> str | None:
    qc = config.get("quantization_config")
    if not isinstance(qc, dict):
        return None
    qc_typed = cast(dict[str, Any], qc)
    method = qc_typed.get("quant_method")
    if method == "fp8":
        return "fp8"
    if method == "fbgemm_fp8":
        return "fbgemm_fp8"
    if method == "finegrained_fp8":
        return "finegrained_fp8"
    if method == "compressed-tensors" and qc_typed.get("format") in (
        "float-quantized",
        "naive-quantized",
    ):
        groups = qc_typed.get("config_groups")
        if isinstance(groups, dict):
            for g in cast(dict[str, Any], groups).values():  # pyright: ignore[reportAny]
                if not isinstance(g, dict):
                    continue
                w = cast(dict[str, Any], g).get("weights")
                if (
                    isinstance(w, dict)
                    and cast(dict[str, Any], w).get("type") == "float"
                    and cast(dict[str, Any], w).get("num_bits") == 8
                ):
                    return "compressed-tensors-fp8"
    return None


def _load_fp8_as_affine8(
    model_path: Path,
    config: dict[str, Any],
    fp8_mode: str,
    strict: bool,
) -> tuple[nn.Module, dict[str, Any]]:
    weight_files = glob.glob(str(model_path / "model*.safetensors"))
    if not weight_files and strict:
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    weights: dict[str, mx.array] = {}
    for wf in weight_files:
        weights.update(cast(dict[str, mx.array], mx.load(wf)))

    weights = _dequantize_fp8_weights(weights, fp8_mode, config)

    stripped_config: dict[str, Any] = {**config}
    stripped_config.pop("quantization_config", None)
    stripped_config.pop("quantization", None)
    text_config = stripped_config.get("text_config")
    if isinstance(text_config, dict):
        text_config_copy: dict[str, Any] = {**cast(dict[str, Any], text_config)}
        text_config_copy.pop("quantization_config", None)
        text_config_copy.pop("quantization", None)
        stripped_config["text_config"] = text_config_copy

    model_class, model_args_class = _get_classes(config=stripped_config)  # pyright: ignore[reportAny]
    model_args = model_args_class.from_dict(stripped_config)  # pyright: ignore[reportAny]
    model = cast(nn.Module, model_class(model_args))

    if hasattr(model, "sanitize"):
        sanitize = cast(
            Callable[[dict[str, mx.array]], dict[str, mx.array]],
            model.sanitize,
        )
        weights = sanitize(weights)

    model.eval()
    model.load_weights(list(weights.items()), strict=strict)

    quant_predicate = _build_quant_predicate(config)
    model, stripped_config = quantize_model(
        model,
        stripped_config,
        group_size=64,
        bits=8,
        mode="affine",
        quant_predicate=quant_predicate,
    )
    return model, stripped_config


def _build_quant_predicate(
    config: dict[str, Any],
) -> Callable[[str, nn.Module], bool] | None:
    qc_raw = config.get("quantization_config")
    if not isinstance(qc_raw, dict):
        return None
    qc = cast(dict[str, Any], qc_raw)
    modules_raw = qc.get("modules_to_not_convert")
    if not isinstance(modules_raw, list):
        return None
    skip_paths: set[str] = set()
    for entry in cast(list[Any], modules_raw):  # pyright: ignore[reportAny]
        if not isinstance(entry, str):
            continue
        mlx_path = _hf_path_to_mlx(entry)
        if mlx_path:
            skip_paths.add(mlx_path)
    if not skip_paths:
        return None

    def predicate(path: str, _module: nn.Module) -> bool:
        for skip in skip_paths:
            if path == skip or path.startswith(skip + "."):
                return False
        return True

    return predicate


def _hf_path_to_mlx(hf_path: str) -> str:
    if hf_path.startswith(("vision_tower", "model.visual")):
        return ""
    if hf_path.startswith("mtp."):
        return ""
    if hf_path.startswith("model.language_model"):
        return hf_path.replace("model.language_model", "language_model.model")
    if hf_path.startswith("language_model."):
        return hf_path
    return "language_model." + hf_path


def _dequantize_fp8_weights(
    weights: dict[str, mx.array],
    fp8_mode: str,
    config: dict[str, Any],
) -> dict[str, mx.array]:
    qc_raw = config.get("quantization_config")
    qc = cast(dict[str, Any], qc_raw) if isinstance(qc_raw, dict) else {}
    block_size = _resolve_block_size(qc)

    out: dict[str, mx.array] = dict(weights)

    for k in list(out.keys()):
        if k.endswith(".weight_scale_inv"):
            scale_inv = out.pop(k)
            weight_key = k.removesuffix("_scale_inv")
            if weight_key in out:
                out[weight_key] = _dequant_blockwise(
                    out[weight_key], scale_inv, block_size
                )
        elif k.endswith(".weight_scale"):
            scale = out.pop(k)
            weight_key = k.removesuffix("_scale")
            if weight_key in out:
                out[weight_key] = _dequant_per_tensor_or_channel(out[weight_key], scale)

    return out


def _resolve_block_size(qc: dict[str, Any]) -> tuple[int, int]:
    block = qc.get("weight_block_size")
    if isinstance(block, (list, tuple)) and len(cast(list[Any], block)) == 2:
        block_typed = cast(list[Any], block)
        a = block_typed[0]  # pyright: ignore[reportAny]
        b = block_typed[1]  # pyright: ignore[reportAny]
        if isinstance(a, int) and isinstance(b, int):
            return a, b
    return 128, 128


def _dequant_per_tensor_or_channel(weight_fp8: mx.array, scale: mx.array) -> mx.array:
    w = cast(mx.array, mx.from_fp8(weight_fp8, dtype=mx.bfloat16))  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
    if scale.ndim == 0:
        return w * scale
    if scale.ndim == 1:
        return w * scale.reshape(-1, 1)
    return w * scale


def _dequant_blockwise(
    weight_fp8: mx.array, scale_inv: mx.array, block_size: tuple[int, int]
) -> mx.array:
    w = cast(mx.array, mx.from_fp8(weight_fp8, dtype=mx.bfloat16))  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
    if w.ndim != 2:
        return w * scale_inv
    bs_m, bs_n = block_size
    m, n = w.shape
    pad_m = (-m) % bs_m
    pad_n = (-n) % bs_n
    pad_width: list[tuple[int, int]] = cast(
        list[tuple[int, int]], cast(object, ((0, pad_m), (0, pad_n)))
    )
    w = mx.pad(w, pad_width)
    w = w.reshape((m + pad_m) // bs_m, bs_m, (n + pad_n) // bs_n, bs_n)
    w = (w * scale_inv[:, None, :, None]).reshape(m + pad_m, n + pad_n)
    return w[:m, :n].astype(mx.bfloat16)
