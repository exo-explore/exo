from __future__ import annotations

from collections.abc import Sequence
from typing import final

import anyio
import pytest

from exo.backends.tinygrad_memory import (
    memory_usage_for_device,
    parse_nvidia_smi_memory,
    parse_rocm_smi_vram,
)
from exo.shared.types.profiling import MemoryUsage
from exo.utils.channels import channel
from exo.utils.info_gatherer.info_gatherer import (
    GatheredInfo,
    InfoGatherer,
    current_memory_usage,
)

_SYSTEM_RAM_TOTAL = 64 * 1024 * 1024 * 1024
_SYSTEM_RAM_AVAILABLE = 60 * 1024 * 1024 * 1024
_SWAP_TOTAL = 8 * 1024 * 1024 * 1024
_SWAP_FREE = 8 * 1024 * 1024 * 1024
_ROCM_TOTAL_BYTES = 17_163_091_968
_ROCM_USED_BYTES = 1_073_741_824
_ROCM_TEXT = (
    "GPU[0] \t\t: VRAM Total Memory (B): 17163091968\n"
    "GPU[0] \t\t: VRAM Total Used Memory (B): 1073741824\n"
    "GPU[1] \t\t: VRAM Total Memory (B): 1\n"
    "GPU[1] \t\t: VRAM Total Used Memory (B): 1\n"
)
_NVIDIA_TOTAL_MEBIBYTES = 16384
_NVIDIA_FREE_MEBIBYTES = 12000
_NVIDIA_TEXT = "16384, 12000\n8192, 100\n"
_MEBIBYTE = 1024 * 1024


@final
class _VirtualMemory:
    def __init__(self, total: int, available: int) -> None:
        self.total = total
        self.available = available


@final
class _SwapMemory:
    def __init__(self, total: int, free: int) -> None:
        self.total = total
        self.free = free


def _patch_host_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    def virtual_memory() -> _VirtualMemory:
        return _VirtualMemory(_SYSTEM_RAM_TOTAL, _SYSTEM_RAM_AVAILABLE)

    def swap_memory() -> _SwapMemory:
        return _SwapMemory(_SWAP_TOTAL, _SWAP_FREE)

    monkeypatch.setattr(
        "exo.shared.types.profiling.psutil.virtual_memory", virtual_memory
    )
    monkeypatch.setattr("exo.shared.types.profiling.psutil.swap_memory", swap_memory)
    monkeypatch.delenv("OVERRIDE_MEMORY_MB", raising=False)


def test_rocm_smi_text_maps_to_bytes() -> None:
    parsed = parse_rocm_smi_vram(_ROCM_TEXT)
    assert parsed == (_ROCM_TOTAL_BYTES, _ROCM_USED_BYTES)


def test_nvidia_smi_text_maps_to_bytes() -> None:
    parsed = parse_nvidia_smi_memory(_NVIDIA_TEXT)
    assert parsed == (
        _NVIDIA_TOTAL_MEBIBYTES * _MEBIBYTE,
        _NVIDIA_FREE_MEBIBYTES * _MEBIBYTE,
    )


def test_unparsed_accelerator_text_is_rejected() -> None:
    assert parse_rocm_smi_vram("no memory fields") is None
    assert parse_nvidia_smi_memory("total, free\n") is None


def test_amd_probe_uses_injected_rocm_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_host_memory(monkeypatch)

    def read_command_output(arguments: Sequence[str]) -> str | None:
        assert arguments[0] == "rocm-smi"
        return _ROCM_TEXT

    monkeypatch.setattr(
        "exo.backends.tinygrad_memory.read_command_output", read_command_output
    )
    usage = memory_usage_for_device("AMD")
    assert usage.ram_total.in_bytes == _ROCM_TOTAL_BYTES
    assert usage.ram_available.in_bytes == _ROCM_TOTAL_BYTES - _ROCM_USED_BYTES
    assert usage.swap_total.in_bytes == _SWAP_TOTAL
    assert usage.swap_available.in_bytes == _SWAP_FREE


def test_cuda_probe_uses_injected_nvidia_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_host_memory(monkeypatch)

    def read_command_output(arguments: Sequence[str]) -> str | None:
        assert arguments[0] == "nvidia-smi"
        return _NVIDIA_TEXT

    monkeypatch.setattr(
        "exo.backends.tinygrad_memory.read_command_output", read_command_output
    )
    usage = memory_usage_for_device("CUDA")
    assert usage.ram_total.in_bytes == _NVIDIA_TOTAL_MEBIBYTES * _MEBIBYTE
    assert usage.ram_available.in_bytes == _NVIDIA_FREE_MEBIBYTES * _MEBIBYTE


def test_missing_accelerator_tool_advertises_zero_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_host_memory(monkeypatch)
    monkeypatch.setenv("OVERRIDE_MEMORY_MB", "14000")

    def read_command_output(arguments: Sequence[str]) -> str | None:
        _ = arguments
        return None

    monkeypatch.setattr(
        "exo.backends.tinygrad_memory.read_command_output", read_command_output
    )
    usage = memory_usage_for_device("AMD")
    assert usage.ram_total.in_bytes == 0
    assert usage.ram_available.in_bytes == 0
    assert usage.swap_total.in_bytes == _SWAP_TOTAL


def test_metal_and_cpu_match_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_host_memory(monkeypatch)
    expected = MemoryUsage.from_psutil(override_memory=None)
    assert memory_usage_for_device("METAL") == expected
    assert memory_usage_for_device("CPU") == expected
    assert expected.ram_available.in_bytes == _SYSTEM_RAM_AVAILABLE


def test_successful_probe_honors_memory_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_host_memory(monkeypatch)
    monkeypatch.setenv("OVERRIDE_MEMORY_MB", "14000")

    def read_command_output(arguments: Sequence[str]) -> str | None:
        _ = arguments
        return _ROCM_TEXT

    monkeypatch.setattr(
        "exo.backends.tinygrad_memory.read_command_output", read_command_output
    )
    usage = memory_usage_for_device("AMD")
    assert usage.ram_total.in_bytes == _ROCM_TOTAL_BYTES
    assert usage.ram_available.in_bytes == 14000 * _MEBIBYTE


def test_gatherer_publishes_vram_when_amd_is_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_host_memory(monkeypatch)
    monkeypatch.setenv("EXO_TINYGRAD_DEVICES", "AMD")

    def read_command_output(arguments: Sequence[str]) -> str | None:
        _ = arguments
        return _ROCM_TEXT

    monkeypatch.setattr(
        "exo.backends.tinygrad_memory.read_command_output", read_command_output
    )
    usage = current_memory_usage()
    assert usage.ram_available.in_bytes == _ROCM_TOTAL_BYTES - _ROCM_USED_BYTES
    assert usage.ram_available.in_bytes != _SYSTEM_RAM_AVAILABLE
    assert usage.ram_total.in_bytes == _ROCM_TOTAL_BYTES


def test_gatherer_without_tinygrad_devices_uses_psutil(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_host_memory(monkeypatch)
    monkeypatch.delenv("EXO_TINYGRAD_DEVICES", raising=False)
    assert current_memory_usage() == MemoryUsage.from_psutil(override_memory=None)


async def test_memory_monitor_sends_probed_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_host_memory(monkeypatch)
    monkeypatch.setenv("EXO_TINYGRAD_DEVICES", "AMD")

    def read_command_output(arguments: Sequence[str]) -> str | None:
        _ = arguments
        return _ROCM_TEXT

    monkeypatch.setattr(
        "exo.backends.tinygrad_memory.read_command_output", read_command_output
    )
    sender, receiver = channel[GatheredInfo]()
    gatherer = InfoGatherer(info_sender=sender)
    received: GatheredInfo | None = None

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(
            gatherer._monitor_memory_usage,  # pyright: ignore[reportPrivateUsage]
            30.0,
        )
        received = await receiver.receive()
        task_group.cancel_scope.cancel()

    assert isinstance(received, MemoryUsage)
    assert received.ram_available.in_bytes == _ROCM_TOTAL_BYTES - _ROCM_USED_BYTES
    assert received.ram_total.in_bytes != _SYSTEM_RAM_TOTAL


def test_engine_memory_usage_delegates_to_the_device_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exo.backends.tinygrad_engine import TinygradDeviceName, TinygradEngine

    def assign_device(device_name: TinygradDeviceName) -> TinygradDeviceName:
        return device_name

    def probe(device_name: TinygradDeviceName) -> MemoryUsage:
        assert device_name == "AMD"
        return MemoryUsage.from_bytes(
            ram_total=_ROCM_TOTAL_BYTES,
            ram_available=_ROCM_TOTAL_BYTES - _ROCM_USED_BYTES,
            swap_total=_SWAP_TOTAL,
            swap_available=_SWAP_FREE,
        )

    monkeypatch.setattr(
        "exo.backends.tinygrad_engine.assign_tinygrad_default_device", assign_device
    )
    monkeypatch.setattr("exo.backends.tinygrad_memory.memory_usage_for_device", probe)
    engine = TinygradEngine(device_name="AMD")
    usage = engine.memory_usage()
    assert usage.ram_available.in_bytes == _ROCM_TOTAL_BYTES - _ROCM_USED_BYTES
