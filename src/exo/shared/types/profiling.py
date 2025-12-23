from enum import IntEnum
from typing import Self

import psutil

from exo.shared.types.memory import Memory
from exo.utils.pydantic_ext import CamelCaseModel


class MemoryPressureLevel(IntEnum):
    """Memory pressure levels matching macOS kernel values.

    On Linux, these are derived from PSI (Pressure Stall Information) metrics.
    """

    NORMAL = 1  # System has adequate free memory
    WARN = 2  # Memory becoming constrained, compression/swap starting
    CRITICAL = 4  # Severe pressure, system actively freeing memory


class PSIMetrics(CamelCaseModel):
    """Linux Pressure Stall Information metrics for memory.

    See: https://docs.kernel.org/accounting/psi.html
    """

    some_avg10: float = 0.0  # % time some tasks stalled (last 10s)
    some_avg60: float = 0.0  # % time some tasks stalled (last 60s)
    some_avg300: float = 0.0  # % time some tasks stalled (last 300s)
    full_avg10: float = 0.0  # % time all tasks stalled (last 10s)
    full_avg60: float = 0.0  # % time all tasks stalled (last 60s)
    full_avg300: float = 0.0  # % time all tasks stalled (last 300s)

    def to_pressure_level(self) -> MemoryPressureLevel:
        """Convert PSI metrics to a pressure level.

        Thresholds based on Facebook's production experience:
        - some_avg10 > 10%: Warning (noticeable latency impact)
        - some_avg10 > 25% or full_avg10 > 10%: Critical (severe impact)
        """
        if self.full_avg10 > 10.0 or self.some_avg10 > 25.0:
            return MemoryPressureLevel.CRITICAL
        if self.some_avg10 > 10.0:
            return MemoryPressureLevel.WARN
        return MemoryPressureLevel.NORMAL


class MemoryPerformanceProfile(CamelCaseModel):
    ram_total: Memory
    ram_available: Memory
    swap_total: Memory
    swap_available: Memory

    # Memory pressure metrics
    pressure_level: MemoryPressureLevel = MemoryPressureLevel.NORMAL
    pressure_pct: float = 0.0  # System-wide free memory percentage (macOS)
    psi: PSIMetrics | None = None  # Linux PSI metrics (None on non-Linux)

    @property
    def effective_available(self) -> Memory:
        """Memory safe to allocate without causing pressure.

        When under pressure, returns a conservative estimate to avoid OOM.
        """
        if self.pressure_level == MemoryPressureLevel.CRITICAL:
            # Under critical pressure, report minimal available to avoid placement
            return Memory.from_bytes(0)
        if self.pressure_level == MemoryPressureLevel.WARN:
            # Under warning, be conservative - use only half of reported available
            return Memory.from_bytes(self.ram_available.in_bytes // 2)
        return self.ram_available

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
        psi: PSIMetrics | None = None,
    ) -> Self:
        return cls(
            ram_total=Memory.from_bytes(ram_total),
            ram_available=Memory.from_bytes(ram_available),
            swap_total=Memory.from_bytes(swap_total),
            swap_available=Memory.from_bytes(swap_available),
            pressure_level=pressure_level,
            pressure_pct=pressure_pct,
            psi=psi,
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


class SystemPerformanceProfile(CamelCaseModel):
    # TODO: flops_fp16: float

    gpu_usage: float = 0.0
    temp: float = 0.0
    sys_power: float = 0.0
    pcpu_usage: float = 0.0
    ecpu_usage: float = 0.0
    ane_power: float = 0.0


class NetworkInterfaceInfo(CamelCaseModel):
    name: str
    ip_address: str


class NodePerformanceProfile(CamelCaseModel):
    model_id: str
    chip_id: str
    friendly_name: str
    memory: MemoryPerformanceProfile
    network_interfaces: list[NetworkInterfaceInfo] = []
    system: SystemPerformanceProfile


class ConnectionProfile(CamelCaseModel):
    throughput: float
    latency: float
    jitter: float
