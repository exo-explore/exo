import shutil
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Literal, Self

import psutil

from exo.shared.types.memory import Memory
from exo.shared.types.thunderbolt import ThunderboltIdentifier
from exo.utils.pydantic_ext import FrozenModel


class MemoryPressureLevel(str, Enum):
    """Memory pressure levels for adaptive placement.

    Implements official TODO.md #13: Memory pressure instead of memory used.
    """

    NORMAL = "normal"      # Comfortable, file caches purgeable (macOS level 1)
    WARNING = "warning"    # Elevated pressure, compression/swapping started (macOS level 2)
    CRITICAL = "critical"  # Severe pressure, thrashing or imminent OOM (macOS level 4)
    UNKNOWN = "unknown"


class MemoryUsage(FrozenModel):
    ram_total: Memory
    ram_available: Memory
    swap_total: Memory
    swap_available: Memory
    pressure_level: MemoryPressureLevel = MemoryPressureLevel.NORMAL
    pressure_pct: float = 0.0

    @classmethod
    def from_bytes(
        cls,
        *,
        ram_total: int,
        ram_available: int,
        swap_total: int,
        swap_available: int,
        pressure_level: MemoryPressureLevel = MemoryPressureLevel.NORMAL,
        pressure_pct: float = 0.0,
    ) -> Self:
        return cls(
            ram_total=Memory.from_bytes(ram_total),
            ram_available=Memory.from_bytes(ram_available),
            swap_total=Memory.from_bytes(swap_total),
            swap_available=Memory.from_bytes(swap_available),
            pressure_level=pressure_level,
            pressure_pct=pressure_pct,
        )

    @classmethod
    def from_psutil(
        cls,
        *,
        override_memory: int | None,
        pressure_level: MemoryPressureLevel = MemoryPressureLevel.NORMAL,
        pressure_pct: float = 0.0,
    ) -> Self:
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()

        return cls.from_bytes(
            ram_total=vm.total,
            ram_available=vm.available if override_memory is None else override_memory,
            swap_total=sm.total,
            swap_available=sm.free,
            pressure_level=pressure_level,
            pressure_pct=pressure_pct,
        )


class DiskUsage(FrozenModel):
    """Disk space usage for the models directory."""

    total: Memory
    available: Memory

    @classmethod
    def from_path(cls, path: Path) -> Self:
        """Get disk usage stats for the partition containing path."""
        total, _used, free = shutil.disk_usage(path)
        return cls(
            total=Memory.from_bytes(total),
            available=Memory.from_bytes(free),
        )


class SystemPerformanceProfile(FrozenModel):
    # TODO: flops_fp16: float

    gpu_usage: float = 0.0
    temp: float = 0.0
    sys_power: float = 0.0
    pcpu_usage: float = 0.0
    ecpu_usage: float = 0.0


InterfaceType = Literal["wifi", "ethernet", "maybe_ethernet", "thunderbolt", "unknown"]


class NetworkInterfaceInfo(FrozenModel):
    name: str
    ip_address: str
    interface_type: InterfaceType = "unknown"


class NodeIdentity(FrozenModel):
    """Static and slow-changing node identification data."""

    model_id: str = "Unknown"
    chip_id: str = "Unknown"
    friendly_name: str = "Unknown"
    os_version: str = "Unknown"
    os_build_version: str = "Unknown"


class NodeNetworkInfo(FrozenModel):
    """Network interface information for a node."""

    interfaces: Sequence[NetworkInterfaceInfo] = []


class NodeThunderboltInfo(FrozenModel):
    """Thunderbolt interface identifiers for a node."""

    interfaces: Sequence[ThunderboltIdentifier] = []


class NodeRdmaCtlStatus(FrozenModel):
    """Whether RDMA is enabled on this node (via rdma_ctl)."""

    enabled: bool


class ThunderboltBridgeStatus(FrozenModel):
    """Whether the Thunderbolt Bridge network service is enabled on this node."""

    enabled: bool
    exists: bool
    service_name: str | None = None
