"""
Compacting event log that reduces memory usage by keeping only the latest
version of events with "replace by key" semantics.

High-frequency events like NodeGatheredInfo, NodeDownloadProgress, and
RunnerStatusUpdated are compacted - only the latest event per unique key
is retained. Critical events (tasks, instances, topology) are preserved.
"""

from collections.abc import Iterator

from exo.shared.types.events import (
    Event,
    NodeDownloadProgress,
    NodeGatheredInfo,
    RunnerStatusUpdated,
)


class CompactingEventLog:
    """
    An event log that compacts high-frequency events while preserving indices.

    Events with "replace by key" semantics (e.g., NodeGatheredInfo per node_id)
    are compacted - only the latest version is retained. The original indices
    are preserved for RequestEventLog compatibility.

    Compacted events:
        - NodeGatheredInfo: latest per node_id
        - NodeDownloadProgress: latest per (node_id, model_id)
        - RunnerStatusUpdated: latest per runner_id

    All other events are preserved in full.
    """

    __slots__ = ("_events", "_compactable_indices", "_compact_count")

    def __init__(self) -> None:
        # Events indexed by their position (None = compacted away)
        self._events: list[Event | None] = []
        # Maps compact_key -> index for quick lookup of previous compactable events
        self._compactable_indices: dict[str, int] = {}
        # Track how many events have been compacted for stats
        self._compact_count: int = 0

    def append(self, event: Event) -> int:
        """
        Append an event to the log, compacting if applicable.

        Returns the index of the appended event.
        """
        key = self._get_compact_key(event)
        idx = len(self._events)

        if key is not None:
            # This is a compactable event - check if we have a previous one
            if key in self._compactable_indices:
                old_idx = self._compactable_indices[key]
                self._events[old_idx] = None  # Mark old entry as compacted
                self._compact_count += 1
            self._compactable_indices[key] = idx

        self._events.append(event)
        return idx

    def get(self, idx: int) -> Event | None:
        """Get event at index, returns None if compacted or out of range."""
        if 0 <= idx < len(self._events):
            return self._events[idx]
        return None

    def iter_from(self, since_idx: int, limit: int = 1000) -> Iterator[tuple[int, Event]]:
        """
        Iterate events from since_idx, skipping compacted entries.

        Yields (index, event) tuples for up to `limit` non-None events.
        """
        count = 0
        for i in range(since_idx, len(self._events)):
            if count >= limit:
                break
            event = self._events[i]
            if event is not None:
                yield i, event
                count += 1

    def __len__(self) -> int:
        """Return total number of indices (including compacted slots)."""
        return len(self._events)

    def __iter__(self) -> Iterator[Event]:
        """Iterate over all non-compacted events."""
        for event in self._events:
            if event is not None:
                yield event

    def active_count(self) -> int:
        """Return count of non-compacted events."""
        return len(self._events) - self._compact_count

    def compacted_count(self) -> int:
        """Return count of compacted (removed) events."""
        return self._compact_count

    def to_list(self) -> list[Event]:
        """Return list of all non-compacted events (for /events endpoint)."""
        return [e for e in self._events if e is not None]

    @staticmethod
    def _get_compact_key(event: Event) -> str | None:
        """
        Get the compaction key for an event, or None if not compactable.

        Compactable events have "replace by key" semantics where only
        the latest value matters for state reconstruction.
        """
        match event:
            case NodeGatheredInfo(node_id=nid):
                return f"ngi:{nid}"
            case NodeDownloadProgress(download_progress=dp):
                # Key by node + model since each (node, model) pair has independent progress
                return f"ndp:{dp.node_id}:{dp.shard_metadata.model_card.model_id}"
            case RunnerStatusUpdated(runner_id=rid):
                return f"rsu:{rid}"
            case _:
                return None
