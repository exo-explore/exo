import csv
import ctypes
import os
import subprocess
import sys
from collections.abc import Callable
from typing import cast

import psutil
from loguru import logger

from exo.shared.types.backends import Backend
from exo.shared.types.memory import Memory
from exo.shared.types.tensorfold import NativeCapabilities
from exo.worker.engines.tensorfold.capabilities import probe_binary


def probe_tensorfold() -> NativeCapabilities | None:
    try:
        capabilities = probe_binary()
    except FileNotFoundError:
        return None
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        logger.warning("TensorFold capability probe failed: {}", exc)
        return None
    return capabilities if capabilities.chip else None


def detected_backends(capabilities: NativeCapabilities | None) -> list[Backend]:
    if capabilities is None or capabilities.chip is None:
        return []
    if (
        sys.platform == "darwin"
        and "metal" in capabilities.backends
        and capabilities.chip.startswith("apple-")
    ):
        return [Backend.TensorFoldMetal]
    if "cuda" in capabilities.backends and capabilities.chip.startswith("nvidia-"):
        return [Backend.TensorFoldCuda]
    return []


def cuda_bus_id() -> str:
    ordinal = int(os.environ.get("TF_CUDA_DEVICE", "0").strip() or "0")
    if ordinal < 0 or ordinal > 2**31 - 1:
        raise ValueError("TF_CUDA_DEVICE is outside the CUDA device range")
    driver = ctypes.CDLL("libcuda.so.1")
    initialize = cast(Callable[[int], int], driver["cuInit"])
    get_device = cast(Callable[[object, int], int], driver["cuDeviceGet"])
    get_bus = cast(Callable[[object, int, int], int], driver["cuDeviceGetPCIBusId"])
    device = ctypes.c_int()
    bus = ctypes.create_string_buffer(32)
    if initialize(0) or get_device(ctypes.byref(device), ordinal):
        raise ValueError("The selected CUDA device is unavailable")
    if get_bus(bus, len(bus), device.value):
        raise ValueError("The selected CUDA device has no PCI bus identity")
    return cast(bytes, bus.value).decode("ascii")


def parse_cuda_memory(output: str, host_available: int) -> Memory | None:
    rows = list(csv.reader(output.strip().splitlines()))
    if len(rows) != 1 or len(rows[0]) != 2:
        return None
    free, name = (value.strip() for value in rows[0])
    if name == "NVIDIA GB10":
        return Memory.from_bytes(max(0, host_available))
    try:
        mib = int(free)
    except ValueError:
        return None
    return Memory.from_bytes(max(0, mib) * 1024**2)


def cuda_memory_available() -> Memory | None:
    try:
        bus_id = cuda_bus_id()
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={bus_id}",
                "--query-gpu=memory.free,name",
                "--format=csv,noheader,nounits",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
        if result.returncode != 0:
            return None
        return parse_cuda_memory(result.stdout, psutil.virtual_memory().available)
    except (OSError, ValueError, AttributeError, subprocess.SubprocessError):
        return None
