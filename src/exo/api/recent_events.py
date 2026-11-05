from collections import deque
from collections.abc import Iterator

from exo.shared.types.events import Event

# GET /events returns up to this many of the most recent events. Every event used to be kept for
# the whole session, growing ~/.exo/event_log without bound.
RECENT_EVENTS = 10_000


class RecentEvents:
    """The most recent events. (Requests' output and inputs, the only large data, never are.)"""

    def __init__(self, max_events: int = RECENT_EVENTS) -> None:
        self._events: deque[Event] = deque(maxlen=max_events)

    def append(self, event: Event) -> None:
        self._events.append(event)

    def __iter__(self) -> Iterator[Event]:
        return iter(self._events)

    def __len__(self) -> int:
        return len(self._events)
