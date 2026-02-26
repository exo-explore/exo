import os
import tomllib
from typing import Literal

import psutil
from pydantic import ConfigDict, Field, ValidationError

from exo.shared.constants import EXO_CONFIG_FILE
from exo.shared.logging import logger
from exo.shared.types.memory import Memory
from exo.utils.pydantic_ext import CamelCaseModel


def _default_memory_threshold() -> float:
    total_gb = Memory.from_bytes(psutil.virtual_memory().total).in_gb
    if total_gb >= 128:
        return 0.85
    if total_gb >= 64:
        return 0.80
    if total_gb >= 32:
        return 0.75
    return 0.70


class MemorySettings(CamelCaseModel):
    model_config = ConfigDict(
        alias_generator=None,
        validate_by_name=True,
        extra="forbid",
        strict=False,
    )

    oom_prevention: bool = False
    memory_threshold: float = Field(default_factory=_default_memory_threshold, ge=0.0, le=1.0)
    memory_floor_gb: float = Field(default=5.0, ge=0.0)


class GenerationSettings(CamelCaseModel):
    model_config = ConfigDict(
        alias_generator=None,
        validate_by_name=True,
        extra="forbid",
        strict=False,
    )

    prefill_step_size: int = Field(default=4096, ge=1)
    max_tokens: int = Field(default=32168, ge=1)
    kv_cache_bits: Literal[4, 8] | None = None


class ExoSettings(CamelCaseModel):
    model_config = ConfigDict(
        alias_generator=None,
        validate_by_name=True,
        extra="ignore",
        strict=False,
    )

    memory: MemorySettings = Field(default_factory=MemorySettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)


_cached_settings: ExoSettings | None = None
_cached_mtime: float = 0.0


def load_settings() -> ExoSettings:
    global _cached_settings, _cached_mtime  # noqa: PLW0603

    try:
        mtime = EXO_CONFIG_FILE.stat().st_mtime
        if _cached_settings is not None and mtime == _cached_mtime:
            return _cached_settings
        with open(EXO_CONFIG_FILE, "rb") as f:
            data = tomllib.load(f)
        settings = ExoSettings.model_validate(data)
        _cached_mtime = mtime
    except FileNotFoundError:
        settings = ExoSettings()
    except (tomllib.TOMLDecodeError, ValidationError) as e:
        logger.warning(f"Invalid config file {EXO_CONFIG_FILE}: {e}")
        settings = ExoSettings()

    # Env vars override config file for backward compat.
    env_threshold = os.environ.get("EXO_MEMORY_THRESHOLD")
    if env_threshold is not None:
        settings = settings.model_copy(
            update={"memory": settings.memory.model_copy(update={"memory_threshold": float(env_threshold)})}
        )
    env_floor = os.environ.get("EXO_MEMORY_FLOOR")
    if env_floor is not None:
        settings = settings.model_copy(
            update={"memory": settings.memory.model_copy(update={"memory_floor_gb": float(env_floor)})}
        )

    _cached_settings = settings
    return settings


def save_settings(settings: ExoSettings) -> None:
    global _cached_settings, _cached_mtime  # noqa: PLW0603

    EXO_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "[memory]",
        f"oom_prevention = {'true' if settings.memory.oom_prevention else 'false'}",
        f"memory_threshold = {settings.memory.memory_threshold}",
        f"memory_floor_gb = {settings.memory.memory_floor_gb}",
        "",
        "[generation]",
        f"prefill_step_size = {settings.generation.prefill_step_size}",
        f"max_tokens = {settings.generation.max_tokens}",
    ]
    if settings.generation.kv_cache_bits is not None:
        lines.append(f"kv_cache_bits = {settings.generation.kv_cache_bits}")

    EXO_CONFIG_FILE.write_text("\n".join(lines) + "\n")

    _cached_settings = settings
    _cached_mtime = EXO_CONFIG_FILE.stat().st_mtime
