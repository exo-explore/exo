import os
import shutil
import subprocess
from pathlib import Path

from pydantic import Field

from exo.shared.types.tensorfold import NativeCapabilities as NativeCapabilities
from exo.shared.types.tensorfold import NativeModel


def resolve_binary(configured: str | None = None) -> Path:
    name = configured or os.environ.get("EXO_TENSORFOLD_BINARY", "tensorfold-native")
    resolved = shutil.which(name)
    if resolved is None:
        raise FileNotFoundError(f"TensorFold executable was not found: {name}")
    return Path(resolved).resolve()


def probe_binary(
    binary: Path | str | None = None, timeout: float = 10.0
) -> NativeCapabilities:
    path = resolve_binary(None if binary is None else str(binary))
    result = subprocess.run(
        [str(path), "capabilities", "--json"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"TensorFold capability probe exited with {result.returncode}")
    if len(result.stdout) > 1024 * 1024:
        raise ValueError("TensorFold capability response exceeds 1 MiB")
    capabilities = NativeCapabilities.model_validate_json(result.stdout)
    capabilities.require_streaming()
    return capabilities


class LoadedCapabilities(NativeModel):
    text_generation: bool
    reasoning: bool = False
    logprobs: bool = False
    penalties: bool = False
    prompt_cache: bool = False
    thinking_switch: bool = False
    thinking_applicable: bool = False
    greedy_only: bool = False


class LoadedModel(NativeModel):
    id: str
    capabilities: LoadedCapabilities
    context_length: int = Field(default=0, ge=0)


class ModelList(NativeModel):
    data: list[LoadedModel]


class HealthMemory(NativeModel):
    peak: int = Field(default=0, ge=0)


class Health(NativeModel):
    status: str
    model: str | None = None
    warming: bool = True
    max_batch_size: int = Field(default=1, ge=1)
    backend: str | None = None
    memory: HealthMemory = Field(default_factory=HealthMemory)
