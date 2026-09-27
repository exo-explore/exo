"""Report device capacity as the ``MemoryUsage`` placement already weights.

``ram_available`` is the memory a shard may occupy: unified RAM for ``METAL``
and ``CPU``, and free VRAM for ``AMD`` and ``CUDA``. The master reads that
field in ``allocate_layers_proportionally`` before a runner exists.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Sequence

from loguru import logger

from exo.backends.tinygrad_engine import TinygradDeviceName
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import MemoryUsage

_ROCM_SMI_ARGUMENTS = ("rocm-smi", "--showmeminfo", "vram")
_NVIDIA_SMI_ARGUMENTS = (
    "nvidia-smi",
    "--query-gpu=memory.total,memory.free",
    "--format=csv,noheader,nounits",
)
_VRAM_TOTAL_BYTES = re.compile(r"VRAM Total Memory \(B\):\s*(\d+)", re.IGNORECASE)
_VRAM_USED_BYTES = re.compile(r"VRAM Total Used Memory \(B\):\s*(\d+)", re.IGNORECASE)
_MEBIBYTE = 1024 * 1024


def read_command_output(arguments: Sequence[str]) -> str | None:
    """Run a memory-query command and return its stdout.

    ``FileNotFoundError``, ``OSError``, and ``subprocess.TimeoutExpired`` are
    handled here and become ``None``. ``memory_usage_for_device`` then
    publishes zero accelerator capacity so placement refuses the node instead
    of treating system RAM as VRAM. The gatherer keeps polling.
    """
    try:
        completed = subprocess.run(
            list(arguments),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def parse_rocm_smi_vram(text: str) -> tuple[int, int] | None:
    """Return ``(total_bytes, used_bytes)`` from ``rocm-smi --showmeminfo vram``.

    The first GPU in the text is the device this node advertises.
    """
    total_match = _VRAM_TOTAL_BYTES.search(text)
    used_match = _VRAM_USED_BYTES.search(text)
    if total_match is None or used_match is None:
        return None
    return int(total_match.group(1)), int(used_match.group(1))


def parse_nvidia_smi_memory(text: str) -> tuple[int, int] | None:
    """Return ``(total_bytes, free_bytes)`` from nounits ``nvidia-smi`` CSV.

    Values are mebibytes. The first GPU row is the device this node advertises.
    """
    for line in text.splitlines():
        columns = [column.strip() for column in line.split(",")]
        if len(columns) < 2:
            continue
        try:
            total_mebibytes = int(columns[0])
            free_mebibytes = int(columns[1])
        except ValueError:
            continue
        return total_mebibytes * _MEBIBYTE, free_mebibytes * _MEBIBYTE
    return None


def _override_available_bytes() -> int | None:
    raw = os.environ.get("OVERRIDE_MEMORY_MB")
    if raw is None or raw.strip() == "":
        return None
    return Memory.from_mb(int(raw)).in_bytes


def _host_swap() -> tuple[int, int]:
    host = MemoryUsage.from_psutil(override_memory=None)
    return host.swap_total.in_bytes, host.swap_available.in_bytes


def _with_override(usage: MemoryUsage) -> MemoryUsage:
    override = _override_available_bytes()
    if override is None:
        return usage
    return MemoryUsage.from_bytes(
        ram_total=usage.ram_total.in_bytes,
        ram_available=override,
        swap_total=usage.swap_total.in_bytes,
        swap_available=usage.swap_available.in_bytes,
    )


def _unavailable_accelerator(device_name: TinygradDeviceName) -> MemoryUsage:
    swap_total, swap_available = _host_swap()
    logger.warning(
        f"Could not read {device_name} memory. Advertising 0 bytes so placement "
        "does not treat system RAM as accelerator memory."
    )
    return MemoryUsage.from_bytes(
        ram_total=0,
        ram_available=0,
        swap_total=swap_total,
        swap_available=swap_available,
    )


def memory_usage_for_device(device_name: TinygradDeviceName) -> MemoryUsage:
    """Return the capacity placement should weight for ``device_name``.

    ``METAL`` and ``CPU`` use ``MemoryUsage.from_psutil``, including
    ``OVERRIDE_MEMORY_MB``. ``AMD`` parses ``rocm-smi``. ``CUDA`` parses
    ``nvidia-smi``. A failed accelerator query publishes zero available bytes
    and does not apply the override, so a missing tool cannot fall back to
    system RAM.
    """
    if device_name == "CPU" or device_name == "METAL":
        return MemoryUsage.from_psutil(override_memory=_override_available_bytes())

    swap_total, swap_available = _host_swap()
    if device_name == "AMD":
        output = read_command_output(_ROCM_SMI_ARGUMENTS)
        parsed = None if output is None else parse_rocm_smi_vram(output)
        if parsed is None:
            return _unavailable_accelerator(device_name)
        total_bytes, used_bytes = parsed
        free_bytes = max(total_bytes - used_bytes, 0)
    else:
        output = read_command_output(_NVIDIA_SMI_ARGUMENTS)
        parsed = None if output is None else parse_nvidia_smi_memory(output)
        if parsed is None:
            return _unavailable_accelerator(device_name)
        total_bytes, free_bytes = parsed

    return _with_override(
        MemoryUsage.from_bytes(
            ram_total=total_bytes,
            ram_available=free_bytes,
            swap_total=swap_total,
            swap_available=swap_available,
        )
    )
