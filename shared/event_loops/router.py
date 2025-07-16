from asyncio.queues import Queue
from typing import Sequence, cast, get_args

from shared.event_loops.main import ExhaustiveMapping
from shared.types.events.common import (
    EventCategories,
    EventCategory,
    EventCategoryEnum,
    EventFromEventLog,
    narrow_event_from_event_log_type,
)

"""
from asyncio import gather
from logging import Logger
from typing import Literal, Protocol, Sequence, TypedDict

from master.sanity_checking import check_keys_in_map_match_enum_values
from shared.types.events.common import EventCategoryEnum
"""

"""
class EventQueues(TypedDict):
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


check_keys_in_map_match_enum_values(EventQueues, EventCategoryEnum)
"""


async def route_events[UnionOfRelevantEvents: EventCategory](
    queue_map: ExhaustiveMapping[
        UnionOfRelevantEvents, Queue[EventFromEventLog[EventCategory]]
    ],
    events: Sequence[EventFromEventLog[EventCategory | EventCategories]],
) -> None:
    """Route an event to the appropriate queue."""
    tuple_of_categories: tuple[EventCategoryEnum, ...] = get_args(UnionOfRelevantEvents)
    print(tuple_of_categories)
    for event in events:
        if isinstance(event.event.event_category, EventCategoryEnum):
            category: EventCategory = event.event.event_category
            if category not in tuple_of_categories:
                continue
            narrowed_event = narrow_event_from_event_log_type(event, category)
            q1: Queue[EventFromEventLog[EventCategory]] = queue_map[
                cast(UnionOfRelevantEvents, category)
            ]  # TODO: make casting unnecessary
            await q1.put(narrowed_event)
        else:
            for category in event.event.event_category:
                if category not in tuple_of_categories:
                    continue
                narrow_event = narrow_event_from_event_log_type(event, category)
                q2 = queue_map[
                    cast(UnionOfRelevantEvents, category)
                ]  # TODO: make casting unnecessary
                await q2.put(narrow_event)
