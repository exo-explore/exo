from collections.abc import Sequence
from typing import Self

import psutil

from exo.shared.types.memory import Memory
from exo.shared.types.thunderbolt import TBIdentifier
from exo.utils.pydantic_ext import CamelCaseModel


class MemoryUsage(CamelCaseModel):
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


class SystemPerformanceProfile(CamelCaseModel):
    # TODO: flops_fp16: float

    gpu_usage: float = 0.0
    temp: float = 0.0
    sys_power: float = 0.0
    pcpu_usage: float = 0.0
    ecpu_usage: float = 0.0


class NetworkInterfaceInfo(CamelCaseModel):
    name: str
    ip_address: str


class NodePerformanceProfile(CamelCaseModel):
    model_id: str = "Unknown"
    chip_id: str = "Unknown"
    friendly_name: str = "Unknown"
    memory: MemoryUsage = MemoryUsage.from_bytes(
        ram_total=0, ram_available=0, swap_total=0, swap_available=0
    )
    network_interfaces: Sequence[NetworkInterfaceInfo] = []
    tb_interfaces: Sequence[TBIdentifier] = []
    system: SystemPerformanceProfile = SystemPerformanceProfile()


class ConnectionProfile(CamelCaseModel):
    pass
