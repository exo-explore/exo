from abc import ABC
from collections.abc import Coroutine

import asyncio

from shared.types.events.events import ResourceProfiledEvent
from shared.types.profiling.common import NodePerformanceProfile, MemoryPerformanceProfile, SystemPerformanceProfile

class EventLog:
    def append(self, event: ResourceProfiledEvent) -> None:
        ...

class ResourceCollector(ABC):
    """
    Details a single resource (or resource type) that is being monitored by the resource monitor.
    """
    def __init__(self, name: str):
        self.name = name

    async def collect(self) -> NodePerformanceProfile:
        ...

class SystemResourceCollector(ResourceCollector):
    def __init__(self):
        super().__init__('system')

    async def collect(self) -> SystemPerformanceProfile:
        ...

class MemoryResourceCollector(ResourceCollector):
    def __init__(self):
        super().__init__('memory')

    async def collect(self) -> MemoryPerformanceProfile:
        ...

class ResourceMonitor:
    def __init__(self, event_outbox: EventLog):
        self.event_outbox: EventLog = event_outbox

        self.collectors: list[ResourceCollector] = [
            SystemResourceCollector(),
            MemoryResourceCollector(),
        ]

    async def collect(self) -> list[NodePerformanceProfile]:
        tasks: list[Coroutine[None, None, NodePerformanceProfile]] = [
            collector.collect() for collector in self.collectors
        ]
        return await asyncio.gather(*tasks)

    async def collect_and_publish(self) -> None:
        profiles = await self.collect()
        for profile in profiles:
            self.event_outbox.append(profile.to_event())