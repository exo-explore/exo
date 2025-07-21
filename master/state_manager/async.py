from asyncio import Lock, Queue, Task, create_task
from logging import Logger
from typing import List, Literal, Protocol, TypedDict

from master.logging import (
    StateUpdateEffectHandlerErrorLogEntry,
    StateUpdateErrorLogEntry,
    StateUpdateLoopAlreadyRunningLogEntry,
    StateUpdateLoopNotRunningLogEntry,
    StateUpdateLoopStartedLogEntry,
    StateUpdateLoopStoppedLogEntry,
)
from master.sanity_checking import check_keys_in_map_match_enum_values
from shared.constants import get_error_reporting_message
from shared.logger import log
from shared.types.events.common import (
    Apply,
    EffectHandler,
    EventCategory,
    EventCategoryEnum,
    EventFromEventLog,
    State,
    StateAndEvent,
)


class AsyncStateManager[EventCategoryT: EventCategory](Protocol):
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
            f"{get_error_reporting_message()}"
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


class AsyncStateManagerMapping(TypedDict):
    MutatesTaskState: AsyncStateManager[Literal[EventCategoryEnum.MutatesTaskState]]
    MutatesTaskSagaState: AsyncStateManager[
        Literal[EventCategoryEnum.MutatesTaskSagaState]
    ]
    MutatesTopologyState: AsyncStateManager[
        Literal[EventCategoryEnum.MutatesTopologyState]
    ]
    MutatesRunnerStatus: AsyncStateManager[
        Literal[EventCategoryEnum.MutatesRunnerStatus]
    ]
    MutatesInstanceState: AsyncStateManager[
        Literal[EventCategoryEnum.MutatesInstanceState]
    ]
    MutatesNodePerformanceState: AsyncStateManager[
        Literal[EventCategoryEnum.MutatesNodePerformanceState]
    ]


check_keys_in_map_match_enum_values(AsyncStateManagerMapping, EventCategoryEnum)
