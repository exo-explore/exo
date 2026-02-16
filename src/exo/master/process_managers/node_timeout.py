from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import final

from loguru import logger

from exo.shared.types.common import NodeId
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

        # Detect duplicate nodes: when a node restarts it gets a new peer ID
        # but keeps the same friendly_name.  Force-timeout the stale peer so
        # its orphaned runners/instances are cleaned up immediately rather
        # than waiting for the 30-second timeout.
        stale_duplicates = _find_stale_duplicate_nodes(state)
        for node_id in stale_duplicates:
            logger.info(
                f"Removing duplicate node {node_id} "
                "(same friendly_name as a newer peer)"
            )
            events.append(NodeTimedOut(node_id=node_id))

        for node_id, last_seen in state.last_seen.items():
            if node_id in stale_duplicates:
                continue  # Already handled above
            if now - last_seen > self.timeout:
                logger.info(f"Removing node {node_id} due to inactivity")
                events.append(NodeTimedOut(node_id=node_id))
        return events


def _find_stale_duplicate_nodes(state: State) -> set[NodeId]:
    """Find node IDs that share a friendly_name with a newer peer.

    When a node restarts quickly (before the 30s timeout), the master sees
    two node_ids with the same friendly_name.  The older one (by last_seen)
    is stale — its runners and instances are unreachable.
    """
    # Group node_ids by friendly_name (skip "Unknown" — not yet identified)
    by_name: dict[str, list[NodeId]] = {}
    for node_id, identity in state.node_identities.items():
        name = identity.friendly_name
        if name == "Unknown":
            continue
        by_name.setdefault(name, []).append(node_id)

    stale: set[NodeId] = set()
    for _name, node_ids in by_name.items():
        if len(node_ids) <= 1:
            continue
        # Keep the node with the most recent last_seen; evict the rest
        newest = max(
            node_ids,
            key=lambda nid: state.last_seen.get(nid, datetime.min.replace(tzinfo=timezone.utc)),
        )
        for nid in node_ids:
            if nid != newest:
                stale.add(nid)
    return stale
