from collections import deque
from collections.abc import Iterator

from exo.shared.types.chunks import ImageChunk
from exo.shared.types.events import ChunkGenerated, Event, InputChunkReceived

# GET /events returns up to this many of the most recent events. Every event, including one per
# generated token, used to be kept for the whole session, growing ~/.exo/event_log without bound.
RECENT_EVENTS = 10_000
# ...holding at most this much image data. Image chunks are the only large events, and a few
# requests with large images would otherwise take most of the memory.
RECENT_IMAGE_BYTES = 64 * 1024 * 1024


class RecentEvents:
    """The most recent events, bounded by their number and by the image data they hold."""

    def __init__(
        self, max_events: int = RECENT_EVENTS, max_image_bytes: int = RECENT_IMAGE_BYTES
    ) -> None:
        self._events: deque[Event] = deque()
        self._image_bytes = 0
        self._max_events = max_events
        self._max_image_bytes = max_image_bytes

    def append(self, event: Event) -> None:
        self._events.append(event)
        self._image_bytes += _image_bytes(event)
        while (
            len(self._events) > self._max_events
            or self._image_bytes > self._max_image_bytes
        ):
            self._image_bytes -= _image_bytes(self._events.popleft())

    def __iter__(self) -> Iterator[Event]:
        return iter(self._events)

    def __len__(self) -> int:
        return len(self._events)


def _image_bytes(event: Event) -> int:
    match event:
        case InputChunkReceived(chunk=chunk):
            return len(chunk.data)
        case ChunkGenerated(chunk=ImageChunk() as chunk):
            return len(chunk.data)
        case _:
            return 0
