import asyncio
from abc import ABC, abstractmethod
from collections.abc import Coroutine
from typing import Callable

from exo.shared.types.profiling import (
    MemoryPerformanceProfile,
    SystemPerformanceProfile,
)


class ResourceCollector(ABC):
    @abstractmethod
    async def collect(self) -> SystemPerformanceProfile | MemoryPerformanceProfile: ...


class SystemResourceCollector(ResourceCollector):
    async def collect(self) -> SystemPerformanceProfile: ...


class MemoryResourceCollector(ResourceCollector):
    async def collect(self) -> MemoryPerformanceProfile: ...


class ResourceMonitor:
    data_collectors: list[ResourceCollector]
    effect_handlers: set[
        Callable[[SystemPerformanceProfile | MemoryPerformanceProfile], None]
    ]

    async def _collect(
        self,
    ) -> list[SystemPerformanceProfile | MemoryPerformanceProfile]:
        tasks: list[
            Coroutine[None, None, SystemPerformanceProfile | MemoryPerformanceProfile]
        ] = [collector.collect() for collector in self.data_collectors]
        return await asyncio.gather(*tasks)

    async def collect(self) -> None:
        profiles = await self._collect()
        for profile in profiles:
            for effect_handler in self.effect_handlers:
                effect_handler(profile)
