"""Loader that swaps in the vendored Qwen3.5/3.6 MTP-aware model class.

If the model declares MTP (``text_config.mtp_num_hidden_layers > 0``)
and a usable weight source exists (either a separate ``mtp.safetensors``
sidecar, or ``mtp.*`` keys in the main shards), the loader dispatches
``model_type='qwen3_5'`` / ``'qwen3_5_moe'`` to
:class:`vendor.qwen3_5_mtp.Model` and threads a sidecar-weights callable
into the model so ``sanitize`` can pick it up.

If MTP is not declared, or no weights are available, the loader falls
through to stock ``mlx_lm.utils.load_model`` (which produces stock
``mlx_lm.models.qwen3_5.Model``).

Strict-load is the only mode supported -- random init of MTP weights
would silently regress to the PR #1226 failure mode.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Type

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import utils as _mlx_lm_utils
from mlx_lm.utils import load_model

from .qwen3_5_mtp import (
    Model as MtpModel,
)
from .qwen3_5_mtp import (
    ModelArgs as MtpModelArgs,
)
from .qwen3_5_mtp import (
    MTPWeightsNotFound,
    _classify_mtp_key_set,
    _quantize_mtp_module,
)

_get_classes: Callable[..., Tuple[Type[nn.Module], Type[Any]]] = (
    _mlx_lm_utils._get_classes  # pyright: ignore[reportAttributeAccessIssue]
)
_SUPPORTED_NATIVE_MTP_MODEL_TYPES = frozenset({"qwen3_5", "qwen3_5_moe"})


# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportPrivateUsage=false, reportAttributeAccessIssue=false, reportUnnecessaryTypeIgnoreComment=false
# Loader operates on untyped safetensors / JSON metadata and patches
# mlx-lm classes whose surface is dynamic.


# ---------------------------------------------------------------------------
# Candidate sidecar locations
# ---------------------------------------------------------------------------


_DEFAULT_SIDECAR_CANDIDATES: tuple[str, ...] = (
    "mtp.safetensors",
    "mtp/weights.safetensors",
    "model-mtp.safetensors",
)


def _config_sidecar_filename(config: Dict[str, Any]) -> Optional[str]:
    extra = config.get("mlx_lm_extra_tensors") or {}
    if isinstance(extra, dict):
        name = extra.get("mtp_file")
        if isinstance(name, str) and name:
            return name
    return None


def _resolve_sidecar_path(model_path: Path, config: Dict[str, Any]) -> Optional[Path]:
    """Return the first existing sidecar path or ``None``."""
    configured = _config_sidecar_filename(config)
    if configured is not None:
        candidate = model_path / configured
        if candidate.exists():
            return candidate
    for rel in _DEFAULT_SIDECAR_CANDIDATES:
        candidate = model_path / rel
        if candidate.exists():
            return candidate
    return None


def _embedded_mtp_keys(model_path: Path) -> Tuple[str, ...]:
    """Probe ``model.safetensors.index.json`` for embedded MTP keys.

    Embedded layouts use either the ``mtp.*`` prefix (original HF
    format) or ``language_model.mtp.*`` (oMLX). We treat both as
    embedded and let the sanitizer normalize the namespace.
    """
    index = model_path / "model.safetensors.index.json"
    if not index.exists():
        return ()
    try:
        payload = json.loads(index.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ()
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        return ()
    return tuple(
        sorted(
            str(k)
            for k in weight_map
            if str(k).startswith("mtp.") or str(k).startswith("language_model.mtp.")
        )
    )


def _sidecar_keys(sidecar: Path) -> Tuple[str, ...]:
    try:
        from safetensors import safe_open  # type: ignore[import-not-found]
    except ImportError:
        return ()
    try:
        with safe_open(str(sidecar), framework="numpy") as handle:  # type: ignore[no-untyped-call]
            # safe_open is not iterable; .keys() is the supported API.
            return tuple(sorted(str(k) for k in handle.keys()))  # noqa: SIM118
    except Exception:
        return ()


# ---------------------------------------------------------------------------
# Weight finalization
# ---------------------------------------------------------------------------


_RMSNORM_SUFFIXES: tuple[str, ...] = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "q_norm.weight",
    "k_norm.weight",
    "pre_fc_norm_hidden.weight",
    "pre_fc_norm_embedding.weight",
    "norm.weight",
)


def _strip_mtp_prefix(key: str) -> str:
    if key.startswith("language_model.mtp."):
        return "mtp." + key[len("language_model.mtp.") :]
    return key


def _load_sidecar_weights(sidecar: Path) -> Dict[str, mx.array]:
    """Load all MTP weights from the sidecar with the ``mtp.`` prefix kept."""
    raw = mx.load(str(sidecar))
    out: Dict[str, mx.array] = {}
    for key, value in raw.items():
        normalized = _strip_mtp_prefix(str(key))
        if normalized.startswith("mtp."):
            out[normalized] = value
    return out


# ---------------------------------------------------------------------------
# Public loader entry point
# ---------------------------------------------------------------------------


def _model_declares_mtp(config: Dict[str, Any]) -> bool:
    tcfg = config.get("text_config", config)
    return int(tcfg.get("mtp_num_hidden_layers", 0) or 0) > 0


def load_mtp_model(
    model_path: Path,
    *,
    lazy: bool = False,
    strict: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load a Qwen3.5/3.6 model, attaching native MTP support if available.

    Returns the loaded model and the (possibly updated) config dict,
    matching ``mlx_lm.utils.load_model``'s signature.

    Raises :class:`MTPWeightsNotFound` if MTP is declared but no weight
    source can be found.
    """
    if not strict:
        raise ValueError(
            "load_mtp_model only supports strict=True; non-strict loading "
            "would silently allow random-initialized MTP weights, "
            "reproducing the PR #1226 failure mode."
        )

    with open(model_path / "config.json", encoding="utf-8") as f:
        config = json.load(f)

    if config.get(
        "model_type"
    ) not in _SUPPORTED_NATIVE_MTP_MODEL_TYPES or not _model_declares_mtp(config):
        # No MTP declared (or wrong model type): fall through to stock.
        return load_model(model_path, lazy=lazy, strict=strict)

    sidecar = _resolve_sidecar_path(model_path, config)
    embedded_keys = _embedded_mtp_keys(model_path)
    has_embedded_mtp = bool(embedded_keys)

    if sidecar is None and not has_embedded_mtp:
        candidates_msg = ", ".join(
            (str(_config_sidecar_filename(config) or ""),) + _DEFAULT_SIDECAR_CANDIDATES
        )
        raise MTPWeightsNotFound(
            "Qwen3.5/3.6 model declares MTP but no MTP weights are present in "
            f"{model_path}. Probed: {candidates_msg} and "
            "model.safetensors.index.json for embedded mtp.* / "
            "language_model.mtp.* keys.",
            candidates=(str(sidecar) if sidecar else "",) + _DEFAULT_SIDECAR_CANDIDATES,
        )

    # Decide quantization policy from the available key set (sidecar
    # takes precedence; if absent, use embedded keys).
    keys_for_policy: tuple[str, ...] = ()
    if sidecar is not None:
        keys_for_policy = _sidecar_keys(sidecar)
    if not keys_for_policy and embedded_keys:
        keys_for_policy = tuple(_strip_mtp_prefix(k) for k in embedded_keys)
    policy = _classify_mtp_key_set(keys_for_policy)

    # Pull bits/group_size from explicit MTP quant config if present,
    # otherwise fall back to the main quantization block.
    mtp_quant = config.get("mtplx_mtp_quantization") or {}
    quant_overrides: Dict[str, Dict[str, Any]] = {}
    if isinstance(mtp_quant, dict) and mtp_quant.get("prequantized"):
        bits = int(mtp_quant.get("bits", 8))
        group_size = int(mtp_quant.get("group_size", 64))
        mode = str(mtp_quant.get("mode", "affine"))
    else:
        main_quant = (
            config.get("quantization") or config.get("quantization_config") or {}
        )
        bits = int(main_quant.get("bits", 4))
        group_size = int(main_quant.get("group_size", 64))
        mode = str(main_quant.get("mode", "affine"))
        if isinstance(main_quant, dict):
            quant_overrides = {
                str(key): value
                for key, value in main_quant.items()
                if isinstance(key, str)
                and key.startswith("language_model.mtp.")
                and isinstance(value, dict)
            }

    # Build a sidecar-loader callable; the TextModel.sanitize calls it
    # when it can't find embedded MTP keys.
    sidecar_loader: Optional[Callable[[], Dict[str, mx.array]]] = None
    if sidecar is not None:
        captured = sidecar  # avoid late-binding

        def _loader() -> Dict[str, mx.array]:
            return _load_sidecar_weights(captured)

        sidecar_loader = _loader

    # Custom get_model_classes that returns our MTP-aware classes for
    # qwen3_5 and arms the model instance with the sidecar loader and
    # the MTP quant policy/params before sanitize runs.
    def get_classes(config: Dict[str, Any]) -> Tuple[Type[nn.Module], Type[Any]]:
        cfg = config
        if cfg.get("model_type") not in _SUPPORTED_NATIVE_MTP_MODEL_TYPES:
            return _get_classes(cfg)

        original_init = MtpModel.__init__

        def patched_init(self_: MtpModel, args: MtpModelArgs) -> None:
            original_init(self_, args)
            self_.language_model._mtp_sidecar_loader = sidecar_loader
            # Quantize MTP module per the detected policy BEFORE
            # load_weights runs in mlx_lm.utils.load_model. The main
            # model's per-layer quantization dict in config["quantization"]
            # does not cover the MTP submodule (no entries with
            # 'mtp.*' paths), so we have to do it ourselves.
            mtp_submodule = self_.language_model.model.mtp
            if mtp_submodule is not None and policy != "unquantized":
                _quantize_mtp_module(
                    mtp_submodule,
                    policy=policy,
                    bits=bits,
                    group_size=group_size,
                    mode=mode,
                    quant_overrides=quant_overrides,
                )

        # Use a tiny subclass so we don't permanently mutate MtpModel.
        class _PatchedMtpModel(MtpModel):
            __init__ = patched_init  # type: ignore[assignment]

        return _PatchedMtpModel, MtpModelArgs

    model, updated_config = load_model(
        model_path,
        lazy=lazy,
        strict=True,
        get_model_classes=get_classes,
    )
    return model, updated_config


__all__ = ["load_mtp_model"]
