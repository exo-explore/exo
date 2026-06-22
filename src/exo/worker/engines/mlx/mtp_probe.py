"""MTP (Multi-Token Prediction) weight probing for Qwen3.6 models.

Detects MTP weights across all known distribution formats:
- Original HuggingFace: embedded in main shards with ``mtp.*`` prefix (15 tensors)
- MTPLX quantized: separate ``mtp.safetensors`` file with ``mtp.*`` prefix (29 tensors)
- oMLX quantized: embedded in main shards with ``language_model.mtp.*`` prefix (29 tensors)
- mlx-community: stripped during quantization (0 tensors, unrecoverable)

The probe is called before model loading so the loader can:
1. Inject separate MTP weights into the main weight dict (MTPLX format)
2. Patch ``sanitize()`` to stop stripping MTP tensors
3. Ensure norm weight shifting fires correctly
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Iterable, cast

from loguru import logger

# pyright: reportAny=false
# This module probes safetensors/JSON data which are untyped.


class MtpFormat(Enum):
    """How MTP weights are stored in this model."""

    ORIGINAL_EMBEDDED = auto()
    """Original HuggingFace format: ``mtp.*`` prefix, embedded in main shards (15 tensors, BF16)."""

    MTPLX_SEPARATE_FILE = auto()
    """MTPLX format: ``mtp.*`` prefix, separate ``mtp.safetensors`` file (29 tensors, quantized)."""

    OMLX_EMBEDDED = auto()
    """oMLX format: ``language_model.mtp.*`` prefix, embedded in main shards (29 tensors, quantized)."""

    STRIPPED = auto()
    """MTP weights were stripped during quantization (e.g. mlx-community). Unrecoverable."""


@dataclass(frozen=True)
class MtpProbeResult:
    """Result of probing a model directory for MTP weights."""

    model_declares_mtp: bool
    """Whether ``config.json`` declares ``mtp_num_hidden_layers > 0``."""

    mtp_tensors_found: bool
    """Whether MTP weight tensors were found on disk."""

    mtp_format: MtpFormat | None
    """Detected storage format, or ``None`` if model doesn't declare MTP."""

    mtp_count: int
    """Number of MTP tensors found."""

    mtp_path: str | None
    """Path to MTP weights (either ``mtp.safetensors`` for separate file,
    or description of embedded location)."""

    mtp_tensor_keys: tuple[str, ...]
    """Names of MTP tensor keys found (empty if STRIPPED or not found)."""

    @property
    def is_recoverable(self) -> bool:
        """Whether MTP weights can be loaded and used."""
        return (
            self.mtp_tensors_found
            and self.mtp_format is not None
            and self.mtp_format != MtpFormat.STRIPPED
        )


def probe_mtp_weights(model_path: Path | str) -> MtpProbeResult:
    """Probe a model directory for MTP weights in all known locations.

    Checks in order:
    1. ``config.json`` → ``mlx_lm_extra_tensors.mtp_file`` (MTPLX separate file)
    2. ``model.safetensors.index.json`` → ``mtp.*`` prefix (Original HuggingFace)
    3. ``model.safetensors.index.json`` → ``language_model.mtp.*`` prefix (oMLX)
    4. ``mtp.safetensors`` file on disk (fallback, no config declaration)

    Args:
        model_path: Path to the model directory.

    Returns:
        ``MtpProbeResult`` describing what was found.
    """
    model_dir = Path(model_path)
    config_path = model_dir / "config.json"

    # Step 1: Check config for MTP declaration
    model_declares_mtp = False
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
            tc = cfg.get("text_config", {})
            mtp_layers = tc.get("mtp_num_hidden_layers", 0)
            model_declares_mtp = bool(mtp_layers and mtp_layers > 0)
        except (json.JSONDecodeError, OSError):
            pass

    if not model_declares_mtp:
        return MtpProbeResult(
            model_declares_mtp=False,
            mtp_tensors_found=False,
            mtp_format=None,
            mtp_count=0,
            mtp_path=None,
            mtp_tensor_keys=(),
        )

    # Step 2: Check for MTPLX separate file via mlx_lm_extra_tensors
    try:
        cfg = json.loads(config_path.read_text())
        extra = cfg.get("mlx_lm_extra_tensors", {})
        mtp_file_name = extra.get("mtp_file")
        if mtp_file_name:
            mtp_file = model_dir / mtp_file_name
            if mtp_file.exists():
                keys = _safetensors_keys(mtp_file)
                mtp_keys = tuple(k for k in keys if "mtp." in k)
                if mtp_keys:
                    return MtpProbeResult(
                        model_declares_mtp=True,
                        mtp_tensors_found=True,
                        mtp_format=MtpFormat.MTPLX_SEPARATE_FILE,
                        mtp_count=len(mtp_keys),
                        mtp_path=str(mtp_file),
                        mtp_tensor_keys=mtp_keys,
                    )
    except (json.JSONDecodeError, OSError):
        pass

    # Step 3: Check weight map index for embedded MTP tensors
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        try:
            idx = json.loads(index_path.read_text())
            weight_map = idx.get("weight_map", {})
            all_keys = tuple(weight_map.keys())

            # Check for oMLX prefix first (more specific)
            omlx_keys = tuple(k for k in all_keys if "language_model.mtp." in str(k))
            if omlx_keys:
                return MtpProbeResult(
                    model_declares_mtp=True,
                    mtp_tensors_found=True,
                    mtp_format=MtpFormat.OMLX_EMBEDDED,
                    mtp_count=len(omlx_keys),
                    mtp_path="embedded in main shards (language_model.mtp.* prefix)",
                    mtp_tensor_keys=omlx_keys,
                )

            # Check for original prefix
            orig_keys = tuple(k for k in all_keys if str(k).startswith("mtp."))
            if orig_keys:
                return MtpProbeResult(
                    model_declares_mtp=True,
                    mtp_tensors_found=True,
                    mtp_format=MtpFormat.ORIGINAL_EMBEDDED,
                    mtp_count=len(orig_keys),
                    mtp_path="embedded in main shards (mtp.* prefix)",
                    mtp_tensor_keys=orig_keys,
                )
        except (json.JSONDecodeError, OSError):
            pass

    # Step 4: Fallback — check for mtp.safetensors without config declaration
    mtp_file = model_dir / "mtp.safetensors"
    if mtp_file.exists():
        keys = _safetensors_keys(mtp_file)
        mtp_keys = tuple(k for k in keys if "mtp." in k)
        if mtp_keys:
            return MtpProbeResult(
                model_declares_mtp=True,
                mtp_tensors_found=True,
                mtp_format=MtpFormat.MTPLX_SEPARATE_FILE,
                mtp_count=len(mtp_keys),
                mtp_path=str(mtp_file),
                mtp_tensor_keys=mtp_keys,
            )

    # Step 5: Model declares MTP but no weights found — stripped during quantization
    logger.warning(
        f"Model at {model_dir} declares mtp_num_hidden_layers > 0 but no MTP "
        "weights were found on disk. MTP weights were likely stripped during "
        "quantization (e.g. mlx-community format). This model will produce "
        "gibberish because the norm weight shift depends on MTP detection."
    )
    return MtpProbeResult(
        model_declares_mtp=True,
        mtp_tensors_found=False,
        mtp_format=MtpFormat.STRIPPED,
        mtp_count=0,
        mtp_path=None,
        mtp_tensor_keys=(),
    )


def _safetensors_keys(path: Path) -> tuple[str, ...]:
    """Return tensor keys from a safetensors file without loading data."""
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="numpy") as f:
            return tuple(f.keys())
    except Exception:
        return ()


def load_mtp_weights(model_path: Path | str) -> dict[str, Any] | None:
    """Load MTP weights from a separate file (MTPLX format).

    Only works for MTPLX_SEPARATE_FILE format. For embedded formats
    (Original, oMLX), the MTP weights are already in the main shards
    and will be loaded by ``mlx_lm``'s normal weight loading.

    Args:
        model_path: Path to the model directory.

    Returns:
        Dict of MTP tensor name → array, or ``None`` if not in separate file.
    """
    result = probe_mtp_weights(model_path)
    if result.mtp_format != MtpFormat.MTPLX_SEPARATE_FILE or result.mtp_path is None:
        return None

    try:
        from safetensors import safe_open

        mtp_weights: dict[str, Any] = {}
        with safe_open(result.mtp_path, framework="mlx") as f:
            for key in cast(Iterable[str], f.keys()):
                if "mtp." in key:
                    mtp_weights[key] = f.get_tensor(key)
        return mtp_weights if mtp_weights else None
    except Exception as e:
        logger.warning(f"Failed to load MTP weights from {result.mtp_path}: {e}")
        return None
