import asyncio
from abc import ABC, abstractmethod
from collections.abc import Coroutine
from typing import Callable, List, Set

from shared.types.profiling.common import (
    MemoryPerformanceProfile,
    NodePerformanceProfile,
    SystemPerformanceProfile,
)


class ResourceCollector(ABC):
    """
    Details a single resource (or resource type) that is being monitored by the resource monitor.
    """

    name = str

    @abstractmethod
    async def collect(self) -> NodePerformanceProfile: ...


class SystemResourceCollector(ResourceCollector):
    name = "system"

    @abstractmethod
    async def collect(self) -> SystemPerformanceProfile: ...


class MemoryResourceCollector(ResourceCollector):
    name = "memory"

    @abstractmethod
    async def collect(self) -> MemoryPerformanceProfile: ...


class ResourceMonitor:
    data_collectors: List[ResourceCollector]
    effect_handlers: Set[Callable[[NodePerformanceProfile], None]]

    # Since there's no implementation, this breaks the typechecker.
    # self.collectors: list[ResourceCollector] = [
    #     SystemResourceCollector(),
    #     MemoryResourceCollector(),
    # ]

    async def _collect(self) -> list[NodePerformanceProfile]:
        tasks: list[Coroutine[None, None, NodePerformanceProfile]] = [
            collector.collect() for collector in self.data_collectors
        ]
        return await asyncio.gather(*tasks)

    async def collect(self) -> None:
        profiles = await self._collect()
        for profile in profiles:
            for effect_handler in self.effect_handlers:
                effect_handler(profile)
