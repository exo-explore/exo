from asyncio import Lock, Queue, Task, create_task, gather
from collections.abc import Mapping
from enum import StrEnum
from logging import Logger
from typing import Any, List, Literal, Protocol, Type, TypedDict

from master.logging import (
    StateUpdateEffectHandlerErrorLogEntry,
    StateUpdateErrorLogEntry,
    StateUpdateLoopAlreadyRunningLogEntry,
    StateUpdateLoopNotRunningLogEntry,
    StateUpdateLoopStartedLogEntry,
    StateUpdateLoopStoppedLogEntry,
)
from shared.constants import EXO_ERROR_REPORTING_MESSAGE
from shared.logger import log
from shared.types.events.common import (
    Apply,
    EffectHandler,
    Event,
    EventCategories,
    EventCategory,
    EventCategoryEnum,
    EventFetcherProtocol,
    EventFromEventLog,
    StateAndEvent,
    State,
)


class QueueMapping(TypedDict):
    MutatesTaskState: Queue[
        EventFromEventLog[Literal[EventCategoryEnum.MutatesTaskState]]
    ]
    MutatesControlPlaneState: Queue[
        EventFromEventLog[Literal[EventCategoryEnum.MutatesControlPlaneState]]
    ]
    MutatesDataPlaneState: Queue[
        EventFromEventLog[Literal[EventCategoryEnum.MutatesDataPlaneState]]
    ]
    MutatesInstanceState: Queue[
        EventFromEventLog[Literal[EventCategoryEnum.MutatesInstanceState]]
    ]
    MutatesNodePerformanceState: Queue[
        EventFromEventLog[Literal[EventCategoryEnum.MutatesNodePerformanceState]]
    ]
    MutatesRunnerStatus: Queue[
        EventFromEventLog[Literal[EventCategoryEnum.MutatesRunnerStatus]]
    ]
    MutatesTaskSagaState: Queue[
        EventFromEventLog[Literal[EventCategoryEnum.MutatesTaskSagaState]]
    ]


def check_keys_in_map_match_enum_values[TEnum: StrEnum](
    mapping_type: Type[Mapping[Any, Any]],
    enum: Type[TEnum],
) -> None:
    mapping_keys = set(mapping_type.__annotations__.keys())
    category_values = set(e.value for e in enum)
    assert mapping_keys == category_values, (
        f"StateDomainMapping keys {mapping_keys} do not match EventCategories values {category_values}"
    )


check_keys_in_map_match_enum_values(QueueMapping, EventCategoryEnum)


class AsyncUpdateStateFromEvents[EventCategoryT: EventCategory](Protocol):
    """Protocol for services that manage a specific state domain."""

    _task: Task[None] | None
    _logger: Logger
    _apply: Apply[EventCategoryT]
    _default_effects: List[EffectHandler[EventCategoryT]]
    extra_effects: List[EffectHandler[EventCategoryT]]
    state: State[EventCategoryT]
    queue: Queue[EventFromEventLog[EventCategoryT]]
    lock: Lock

    def __init__(
        self,
        state: State[EventCategoryT],
        queue: Queue[EventFromEventLog[EventCategoryT]],
        extra_effects: List[EffectHandler[EventCategoryT]],
        logger: Logger,
    ) -> None:
        """Initialise the service with its event queue."""
        self.state = state
        self.queue = queue
        self.extra_effects = extra_effects
        self._logger = logger
        self._task = None

    async def read_state(self) -> State[EventCategoryT]:
        """Get a thread-safe snapshot of this service's state domain."""
        return self.state.model_copy(deep=True)

    @property
    def is_running(self) -> bool:
        """Check if the service's event loop is running."""
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        """Start the service's event loop."""
        if self.is_running:
            log(self._logger, StateUpdateLoopAlreadyRunningLogEntry())
            raise RuntimeError("State Update Loop Already Running")
        log(self._logger, StateUpdateLoopStartedLogEntry())
        self._task = create_task(self._event_loop())

    async def stop(self) -> None:
        """Stop the service's event loop."""
        if not self.is_running:
            log(self._logger, StateUpdateLoopNotRunningLogEntry())
            raise RuntimeError("State Update Loop Not Running")

        assert self._task is not None, (
            f"{EXO_ERROR_REPORTING_MESSAGE()}"
            "BUG: is_running is True but _task is None, this should never happen!"
        )
        self._task.cancel()
        log(self._logger, StateUpdateLoopStoppedLogEntry())

    async def _event_loop(self) -> None:
        """Event loop for the service."""
        while True:
            event = await self.queue.get()
            previous_state = self.state.model_copy(deep=True)
            try:
                async with self.lock:
                    updated_state = self._apply(
                        self.state,
                        event,
                    )
                    self.state = updated_state
            except Exception as e:
                log(self._logger, StateUpdateErrorLogEntry(error=e))
                raise e
            try:
                for effect_handler in self._default_effects + self.extra_effects:
                    effect_handler(StateAndEvent(previous_state, event), updated_state)
            except Exception as e:
                log(self._logger, StateUpdateEffectHandlerErrorLogEntry(error=e))
                raise e


class EventRouter:
    """Routes events to appropriate services based on event categories."""

    queue_map: QueueMapping
    event_fetcher: EventFetcherProtocol[EventCategory]
    _logger: Logger

    async def _get_queue_by_category[T: EventCategory](
        self, category: T
    ) -> Queue[Event[T]]:
        """Get the queue for a given category."""
        category_str: str = category.value
        queue: Queue[Event[T]] = self.queue_map[category_str]

    async def _process_events[T: EventCategory](self, category: T) -> None:
        """Process events for a given domain."""
        queue: Queue[Event[T]] = await self._get_queue_by_category(category)
        events_to_process: list[Event[T]] = []
        while not queue.empty():
            events_to_process.append(await queue.get())
        for event_to_process in events_to_process:
            await self.queue_map[category].put(event_to_process)
        return None

    async def _submit_events(
        self, events: list[Event[EventCategory | EventCategories]]
    ) -> None:
        """Route multiple events to their appropriate services."""
        for event in events:
            for category in event.event_category:
                await self._event_queues[category].put(event)

        await gather(
            *[self._process_events(domain) for domain in self._event_queues.keys()]
        )

    async def _get_events_to_process(self) -> list[Event[EventCategories]]:
        """Get events to process from the event fetcher."""
