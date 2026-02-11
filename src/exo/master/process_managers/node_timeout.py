from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import final

from loguru import logger

from exo.shared.types.events import Event, NodeTimedOut
from exo.shared.types.state import State

_DEFAULT_TIMEOUT = timedelta(seconds=30)


@final
class NodeTimeoutReconciler:
    """Time out nodes that haven't been seen recently."""

    def __init__(self, timeout: timedelta = _DEFAULT_TIMEOUT) -> None:
        self.timeout = timeout

    async def reconcile(self, state: State) -> Sequence[Event]:
        now = datetime.now(tz=timezone.utc)
        events: list[Event] = []
        for node_id, last_seen in state.last_seen.items():
            if now - last_seen > self.timeout:
                logger.info(f"Removing node {node_id} due to inactivity")
                events.append(NodeTimedOut(node_id=node_id))
        return events
