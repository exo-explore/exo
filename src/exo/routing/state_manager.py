from collections.abc import Callable, Iterable
from dataclasses import dataclass

import exo.routing.topics as topics
from exo.shared.apply import apply
from exo.shared.types.common import SessionId
from exo.shared.types.events import (
    IndexedEvent,
)
from exo.shared.types.state import BaseState, ForwarderState, State
from exo.utils import fmap, fold
from exo.utils.channels import Receiver

from .event_router import EventRouter
from .router import Router


@dataclass
class StateManager[T: BaseState]:
    _apply: Callable[[T, IndexedEvent], T]
    _event_recv: Receiver[IndexedEvent]
    _state_recv: Receiver[T]
    _state: T

    def get(self) -> T:
        return self._state

    async def run(self):
        async with self._event_recv, self._state_recv:
            async for event in self._event_recv:
                # apply new states eagerly
                def order_state(current: T, other: T) -> T:
                    return (
                        current
                        if other.last_event_idx() < current.last_event_idx()
                        else other
                    )

                self._state = fold(self._state, order_state, self._state_recv.collect())
                # catch up / ignore stale
                if event.idx <= self._state.last_event_idx():
                    continue
                # apply state
                self._state = self._apply(self._state, event)


@dataclass
class _Hack:
    recv: Receiver[ForwarderState]
    session_id: SessionId

    def collect(self) -> Iterable[State]:
        return fmap(self._matches, self.recv.collect())

    def _matches(self, s: ForwarderState) -> State | None:
        return s.state if s.session_id == self.session_id else None


def state_manager_from_routers(
    router: Router, event_router: EventRouter
) -> StateManager[State]:
    return StateManager(
        apply,
        event_router.receiver(),
        _Hack(router.receiver(topics.STATE_SNAPSHOTS), event_router.session_id),  # type: ignore
        State(),
    )
