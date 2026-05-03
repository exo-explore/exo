import shutil
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal, Self

import psutil

from exo.shared.types.memory import Memory
from exo.shared.types.thunderbolt import ThunderboltIdentifier
from exo.utils.pydantic_ext import FrozenModel


class MemoryUsage(FrozenModel):
    ram_total: Memory
    ram_available: Memory
    swap_total: Memory
    swap_available: Memory

    @classmethod
    def from_bytes(
        cls, *, ram_total: int, ram_available: int, swap_total: int, swap_available: int
    ) -> Self:
        return cls(
            ram_total=Memory.from_bytes(ram_total),
            ram_available=Memory.from_bytes(ram_available),
            swap_total=Memory.from_bytes(swap_total),
            swap_available=Memory.from_bytes(swap_available),
        )

    @classmethod
    def from_psutil(cls, *, override_memory: int | None) -> Self:
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()

        return cls.from_bytes(
            ram_total=vm.total,
            ram_available=vm.available if override_memory is None else override_memory,
            swap_total=sm.total,
            swap_available=sm.free,
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


class NodeGpuProfile(FrozenModel):
    """Measured GPU compute throughput and memory bandwidth for a node."""

    engine: Literal["mlx"]
    tflops_fp16: float
    memory_bandwidth_gbps: float
    measured_at: datetime


class NodeSocketLinkProfile(FrozenModel):
    """Per-direction TCP/IP bandwidth + round-trip latency (with jitter).

    `upload_mbps` is source -> sink (this node sending), `download_mbps` is
    sink -> source. `latency_jitter_ms` is the mean of |Δ| between
    consecutive RTT samples (RFC 3550 / iperf3 jitter convention).
    """

    transport: Literal["socket"] = "socket"
    sink_ip: str
    latency_ms: float
    latency_jitter_ms: float = 0.0
    upload_mbps: float
    download_mbps: float
    measured_at: datetime


class NodeRdmaLinkProfile(FrozenModel):
    """Per-direction RDMA bandwidth + round-trip latency (with jitter) over a TB edge.

    All numeric fields are None when the most recent probe was skipped (peer
    busy) or failed. We never substitute synthetic numbers.
    """

    transport: Literal["rdma"] = "rdma"
    source_rdma_iface: str
    sink_rdma_iface: str
    upload_mbps: float | None
    download_mbps: float | None
    payload_bytes: int | None
    latency_ms: float | None = None
    latency_jitter_ms: float | None = None
    measured_at: datetime


NodeLinkProfile = NodeSocketLinkProfile | NodeRdmaLinkProfile
