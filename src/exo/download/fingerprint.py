"""Content-based model fingerprinting.

A fingerprint is a stable hash over the architecture-defining subset of a model's
``config.json``. Two model directories with the same fingerprint represent the same
model variant — same architecture, same hyperparameters, same quantization — even if
they live under different folder names on disk.

Used by :func:`exo.download.download_utils.resolve_existing_model` to find a locally
cached model regardless of how its folder was named, after the convention-based
``{models_dir}/{normalized_id}`` lookup misses.

Pure functions, no side effects. ``config.json`` is the only input.
"""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final

# Architecture-defining fields. Quantization is included so a 4-bit MLX quant doesn't
# match its bf16 sibling (they share most other fields). Cosmetic config keys —
# ``_name_or_path``, ``transformers_version``, ``torch_dtype`` overrides at save time,
# ``architectures`` capitalisation drift — are deliberately excluded so two saves of
# the same model from different transformers versions still match.
_FINGERPRINT_KEYS: Final = (
    "architectures",
    "model_type",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "vocab_size",
    "max_position_embeddings",
    "head_dim",
    "quantization",
)


def fingerprint_config(config: Mapping[str, object]) -> str:
    """Stable SHA-256 over a canonical projection of architecture-defining fields.

    Cosmetic differences (key order, whitespace, unrelated keys) do not affect the
    result. Two configs that share every value listed in ``_FINGERPRINT_KEYS`` produce
    the same fingerprint.
    """
    selected = {k: config[k] for k in _FINGERPRINT_KEYS if k in config}
    canonical = json.dumps(selected, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def fingerprint_directory(model_dir: Path) -> str | None:
    """Return the fingerprint of ``{model_dir}/config.json``, or ``None`` if absent or invalid.

    Tolerates malformed JSON and missing files by returning ``None`` — callers should
    treat that as "this isn't a recognisable model directory" rather than as an error.
    """
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return None
    try:
        with config_path.open("r", encoding="utf-8") as f:
            raw: object = json.load(f)  # pyright: ignore[reportAny]
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    return fingerprint_config(raw)  # pyright: ignore[reportUnknownArgumentType]
