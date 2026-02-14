from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from exo.shared.types.events import Event
from exo.shared.types.state import State


@runtime_checkable
class ProcessManager(Protocol):
    """A reconciliation step that examines state and returns corrective events."""

    async def reconcile(self, state: State) -> Sequence[Event]: ...
