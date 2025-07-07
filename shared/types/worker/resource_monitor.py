import asyncio
from abc import ABC, abstractmethod
from collections.abc import Coroutine
from typing import Callable, Set

from shared.types.events.events import ResourceProfiled
from shared.types.profiling.common import (
    MemoryPerformanceProfile,
    NodePerformanceProfile,
    SystemPerformanceProfile,
)


class EventLog:
    def append(self, event: ResourceProfiled) -> None: ...


class ResourceCollector(ABC):
    """
    Details a single resource (or resource type) that is being monitored by the resource monitor.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def collect(self) -> NodePerformanceProfile: ...


class SystemResourceCollector(ResourceCollector):
    def __init__(self):
        super().__init__("system")

    @abstractmethod
    async def collect(self) -> SystemPerformanceProfile: ...


class MemoryResourceCollector(ResourceCollector):
    def __init__(self):
        super().__init__("memory")

    @abstractmethod
    async def collect(self) -> MemoryPerformanceProfile: ...


class ResourceMonitor:
    def __init__(
        self,
        collectors: list[ResourceCollector],
        effect_handlers: Set[Callable[[NodePerformanceProfile], None]],
    ):
        self.effect_handlers: Set[Callable[[NodePerformanceProfile], None]] = (
            effect_handlers
        )
        self.collectors: list[ResourceCollector] = collectors

        # Since there's no implementation, this breaks the typechecker.
        # self.collectors: list[ResourceCollector] = [
        #     SystemResourceCollector(),
        #     MemoryResourceCollector(),
        # ]

    async def _collect(self) -> list[NodePerformanceProfile]:
        tasks: list[Coroutine[None, None, NodePerformanceProfile]] = [
            collector.collect() for collector in self.collectors
        ]
        return await asyncio.gather(*tasks)

    async def collect(self) -> None:
        profiles = await self._collect()
        for profile in profiles:
            for effect_handler in self.effect_handlers:
                effect_handler(profile)
