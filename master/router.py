from asyncio import Queue, gather
from logging import Logger
from typing import Literal, TypedDict

from master.sanity_checking import check_keys_in_map_match_enum_values
from shared.types.events.common import (
    EventCategories,
    EventCategory,
    EventCategoryEnum,
    EventFetcherProtocol,
    EventFromEventLog,
    narrow_event_from_event_log_type,
)


class QueueMapping(TypedDict):
    MutatesTaskState: Queue[
        EventFromEventLog[Literal[EventCategoryEnum.MutatesTaskState]]
    ]
    MutatesTaskSagaState: Queue[
        EventFromEventLog[Literal[EventCategoryEnum.MutatesTaskSagaState]]
    ]
    MutatesControlPlaneState: Queue[
        EventFromEventLog[Literal[EventCategoryEnum.MutatesControlPlaneState]]
    ]
    MutatesDataPlaneState: Queue[
        EventFromEventLog[Literal[EventCategoryEnum.MutatesDataPlaneState]]
    ]
    MutatesRunnerStatus: Queue[
        EventFromEventLog[Literal[EventCategoryEnum.MutatesRunnerStatus]]
    ]
    MutatesInstanceState: Queue[
        EventFromEventLog[Literal[EventCategoryEnum.MutatesInstanceState]]
    ]
    MutatesNodePerformanceState: Queue[
        EventFromEventLog[Literal[EventCategoryEnum.MutatesNodePerformanceState]]
    ]


check_keys_in_map_match_enum_values(QueueMapping, EventCategoryEnum)


class EventRouter:
    """Routes events to appropriate services based on event categories."""

    queue_map: QueueMapping
    event_fetcher: EventFetcherProtocol[EventCategory]
    _logger: Logger

    async def _get_queue_by_category[T: EventCategory](
        self, category: T
    ) -> Queue[EventFromEventLog[T]]:
        """Get the queue for a given category."""
        category_str: str = category.value
        queue: Queue[EventFromEventLog[T]] = self.queue_map[category_str]
        return queue

    async def _process_events[T: EventCategory](self, category: T) -> None:
        """Process events for a given domain."""
        queue: Queue[EventFromEventLog[T]] = await self._get_queue_by_category(category)
        events_to_process: list[EventFromEventLog[T]] = []
        while not queue.empty():
            events_to_process.append(await queue.get())
        for event_to_process in events_to_process:
            await self.queue_map[category.value].put(event_to_process)
        return None

    async def _submit_events[T: EventCategory | EventCategories](
        self, events: list[EventFromEventLog[T]]
    ) -> None:
        """Route multiple events to their appropriate services."""
        for event in events:
            if isinstance(event.event.event_category, EventCategory):
                q1: Queue[EventFromEventLog[T]] = self.queue_map[
                    event.event.event_category.value
                ]
                await q1.put(event)
            elif isinstance(event.event.event_category, EventCategories):
                for category in event.event.event_category:
                    narrow_event = narrow_event_from_event_log_type(event, category)
                    q2: Queue[EventFromEventLog[T]] = self.queue_map[category.value]
                    await q2.put(narrow_event)

        await gather(*[self._process_events(domain) for domain in EventCategoryEnum])

    async def _get_events_to_process(
        self,
    ) -> list[EventFromEventLog[EventCategories | EventCategory]]:
        """Get events to process from the event fetcher."""
