"""In-memory store for OpenAI Responses streaming events.

This supports:
- recording streaming events per response id
- replaying events starting after a given sequence_number
- observing completion / cancellation status
"""

from __future__ import annotations

from collections import defaultdict
from typing import AsyncGenerator, Dict, List, Optional

import anyio

from exo.shared.types.openai_responses import ResponsesStreamEvent


class _StoredResponse:
    def __init__(self) -> None:
        self.events: List[ResponsesStreamEvent] = []
        self.status: str = "in_progress"  # in_progress | completed | cancelled
        self._new_event: anyio.Event = anyio.Event()

    def append(self, event: ResponsesStreamEvent) -> None:
        self.events.append(event)
        # Mark completed when we see the terminal event
        if event.type == "response.completed":
            self.status = "completed"
        self._new_event.set()

    async def wait_for_new_events(self, current_len: int) -> None:
        """Block until events list grows beyond current_len or status is terminal."""
        if self.status in ("completed", "cancelled"):
            return
        if len(self.events) > current_len:
            return
        await self._new_event.wait()
        # Reset for next waiter
        self._new_event = anyio.Event()


class ResponsesStore:
    """Simple per-process store of Responses streaming events."""

    def __init__(self) -> None:
        self._store: Dict[str, _StoredResponse] = {}

    def record_event(self, response_id: str, event: ResponsesStreamEvent) -> None:
        """Record a streaming event for a response."""
        stored = self._store.setdefault(response_id, _StoredResponse())
        stored.append(event)

    def mark_cancelled(self, response_id: str) -> None:
        """Mark a response as cancelled."""
        stored = self._store.setdefault(response_id, _StoredResponse())
        stored.status = "cancelled"
        stored._new_event.set()

    def get_status(self, response_id: str) -> Optional[str]:
        stored = self._store.get(response_id)
        return stored.status if stored is not None else None

    async def stream_from(
        self, response_id: str, starting_after: int
    ) -> AsyncGenerator[ResponsesStreamEvent, None]:
        """Yield events for a response, starting after a given sequence_number.

        This will replay already-recorded events with sequence_number > starting_after,
        then continue to yield newly-recorded events until the response is completed
        or cancelled.
        """
        stored = self._store.get(response_id)
        if stored is None:
            return

        # Find first index whose sequence_number is greater than starting_after
        idx = 0
        for i, ev in enumerate(stored.events):
            if ev.sequence_number > starting_after:
                idx = i
                break
        else:
            idx = len(stored.events)

        while True:
            # Yield any events we haven't seen yet
            while idx < len(stored.events):
                ev = stored.events[idx]
                idx += 1
                if ev.sequence_number > starting_after:
                    yield ev

            if stored.status in ("completed", "cancelled"):
                break

            await stored.wait_for_new_events(len(stored.events))

