from asyncio import Lock, Task
from asyncio import Queue as AsyncQueue
from collections.abc import MutableMapping
from logging import Logger
from typing import Any, Hashable, Mapping, Protocol, Sequence

from fastapi.responses import Response, StreamingResponse

from shared.event_loops.commands import ExternalCommand
from shared.types.events.components import Apply, EventFromEventLog
from shared.types.events.registry import Event
from shared.types.state import State


class ExhaustiveMapping[K: Hashable, V](MutableMapping[K, V]):
    __slots__ = ("_store",)

    required_keys: frozenset[K] = frozenset()

    def __init__(self, data: Mapping[K, V]):
        missing = self.required_keys - data.keys()
        extra = data.keys() - self.required_keys
        if missing or extra:
            raise ValueError(f"missing={missing!r}, extra={extra!r}")
        self._store: dict[K, V] = dict(data)

    def __getitem__(self, k: K) -> V:
        return self._store[k]

    def __setitem__(self, k: K, v: V) -> None:
        self._store[k] = v

    def __delitem__(self, k: K) -> None:
        del self._store[k]

    def __iter__(self):
        return iter(self._store)

    def __len__(self) -> int:
        return len(self._store)


def apply_events(
    state: State, apply_fn: Apply, events: Sequence[EventFromEventLog[Event]]
) -> State:
    sorted_events = sorted(events, key=lambda event: event.idx_in_log)
    state = state.model_copy()
    for wrapped_event in sorted_events:
        if wrapped_event.idx_in_log <= state.last_event_applied_idx:
            continue
        state.last_event_applied_idx = wrapped_event.idx_in_log
        state = apply_fn(state, wrapped_event.event)
    return state


class NodeCommandLoopProtocol(Protocol):
    _command_runner: Task[Any] | None = None
    _command_queue: AsyncQueue[ExternalCommand]
    _response_queue: AsyncQueue[Response | StreamingResponse]
    _logger: Logger

    @property
    def is_command_runner_running(self) -> bool:
        return self._command_runner is not None and not self._command_runner.done()

    async def start_command_runner(self) -> None: ...
    async def stop_command_runner(self) -> None: ...
    async def push_command(self, command: ExternalCommand) -> None: ...
    async def pop_response(self) -> Response | StreamingResponse: ...
    async def _handle_command(self, command: ExternalCommand) -> None: ...


class NodeEventGetterProtocol(Protocol):
    _event_fetcher: Task[Any] | None = None
    _event_queue: AsyncQueue[EventFromEventLog[Event]]
    _logger: Logger

    @property
    async def is_event_fetcher_running(self) -> bool:
        return self._event_fetcher is not None and not self._event_fetcher.done()

    async def start_event_fetcher(self) -> None: ...
    async def stop_event_fetcher(self) -> None: ...


class NodeStateStorageProtocol(Protocol):
    _state: State
    _state_lock: Lock
    _logger: Logger

    async def _read_state(
        self,
    ) -> State: ...


class NodeStateManagerProtocol(
    NodeEventGetterProtocol, NodeStateStorageProtocol
):
    _state_manager: Task[Any] | None = None
    _logger: Logger

    @property
    async def is_state_manager_running(self) -> bool:
        is_task_running = (
            self._state_manager is not None and not self._state_manager.done()
        )
        return (
            is_task_running
            and await self.is_event_fetcher_running
            and await self.is_state_manager_running
        )

    async def start_state_manager(self) -> None: ...
    async def stop_state_manager(self) -> None: ...
    async def _apply_queued_events(self) -> None: ...


class NodeEventLoopProtocol(
    NodeCommandLoopProtocol, NodeStateManagerProtocol
): ...
