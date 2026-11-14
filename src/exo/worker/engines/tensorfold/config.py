import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from exo.worker.engines.tensorfold.capabilities import resolve_binary


def validate_single_rank_environment(environment: Mapping[str, str]) -> None:
    for family in ("GLM", "NEMOTRON", "MINIMAX", "TP", "STAGE"):
        for setting, expected in (("WORLD", "1"), ("RANK", "0")):
            name = f"TF_{family}_{setting}"
            if environment.get(name) not in (None, expected):
                raise ValueError(
                    f"TensorFold's EXO runtime requires {name} to be unset or {expected}"
                )


def positive_seconds(name: str, default: float) -> float:
    value = float(os.environ.get(name, str(default)))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


@dataclass(frozen=True)
class NativeSettings:
    binary: Path
    startup_timeout: float = 900.0
    request_timeout: float = 300.0
    shutdown_timeout: float = 5.0

    @classmethod
    def from_environment(cls) -> "NativeSettings":
        return cls(
            binary=resolve_binary(),
            startup_timeout=positive_seconds("EXO_TENSORFOLD_STARTUP_TIMEOUT", 900.0),
            request_timeout=positive_seconds("EXO_TENSORFOLD_REQUEST_TIMEOUT", 300.0),
            shutdown_timeout=positive_seconds("EXO_TENSORFOLD_SHUTDOWN_TIMEOUT", 5.0),
        )
