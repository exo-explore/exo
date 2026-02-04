from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass(frozen=True)
class ModelFingerprint:
    """Lightweight fingerprint derived from local `config.json`."""

    model_type: str | None = None
    architectures: tuple[str, ...] = ()
    tokenizer_class: str | None = None


def load_model_fingerprint(model_path: Path) -> ModelFingerprint:
    """Parse `config.json` from the local model directory.

    Some models embed relevant fields under `text_config`. We prefer those values
    when present to avoid special-casing at call sites.
    """

    config_path = model_path / "config.json"
    try:
        raw = json.loads(config_path.read_text())
    except FileNotFoundError:
        return ModelFingerprint()
    except Exception as e:
        logger.debug(f"Failed to parse config.json at {config_path}: {e}")
        return ModelFingerprint()

    if not isinstance(raw, dict):
        return ModelFingerprint()

    data: dict[str, Any] = dict(raw)
    text_config = data.get("text_config")
    if isinstance(text_config, dict):
        for key in ("architectures", "model_type", "tokenizer_class"):
            if (val := text_config.get(key)) is not None:
                data[key] = val

    model_type = data.get("model_type")
    if isinstance(model_type, str):
        model_type = model_type.strip() or None
    else:
        model_type = None

    tokenizer_class = data.get("tokenizer_class")
    if isinstance(tokenizer_class, str):
        tokenizer_class = tokenizer_class.strip() or None
    else:
        tokenizer_class = None

    arch_raw = data.get("architectures")
    architectures: tuple[str, ...] = ()
    if isinstance(arch_raw, list):
        architectures = tuple(a for a in arch_raw if isinstance(a, str))

    return ModelFingerprint(
        model_type=model_type,
        architectures=architectures,
        tokenizer_class=tokenizer_class,
    )


def is_kimi_tokenizer_repo(model_path: Path) -> bool:
    """Detect Kimi-style repos by the presence of custom tokenizer code."""

    return (model_path / "tokenization_kimi.py").exists()


def is_gemma3(fingerprint: ModelFingerprint) -> bool:
    model_type = (fingerprint.model_type or "").lower()
    if model_type.startswith("gemma3"):
        return True
    return any("gemma3" in arch.lower() for arch in fingerprint.architectures)


def is_glm(fingerprint: ModelFingerprint) -> bool:
    model_type = (fingerprint.model_type or "").lower()
    if model_type.startswith("glm"):
        return True
    # GLM model architectures commonly use Glm* or ChatGLM* naming.
    return any(
        arch.lower().startswith("glm") or "chatglm" in arch.lower()
        for arch in fingerprint.architectures
    )

