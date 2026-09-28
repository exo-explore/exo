"""The API remembers only the most recent events, so its history can't grow without bound."""

from collections import deque
from typing import cast

from fastapi import FastAPI
from fastapi.testclient import TestClient

from exo.api.main import API, RECENT_EVENTS
from exo.shared.types.events import Event, IndexedEvent, TestEvent
from exo.shared.types.state import State
from exo.utils.channels import channel


async def test_events_endpoint_returns_the_most_recent_events() -> None:
    events = [TestEvent() for _ in range(RECENT_EVENTS + 5)]
    send, recv = channel[IndexedEvent]()
    api = object.__new__(API)
    api.state = State()
    api.event_receiver = recv
    api._recent_events = deque[Event](maxlen=RECENT_EVENTS)  # pyright: ignore[reportPrivateUsage]
    api._text_generation_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._image_generation_queues = {}  # pyright: ignore[reportPrivateUsage]
    for idx, event in enumerate(events):
        await send.send(IndexedEvent(idx=idx, event=event))
    send.close()

    await api._apply_state()  # pyright: ignore[reportPrivateUsage]

    app = FastAPI()
    app.get("/events")(api.stream_events)
    returned = cast(
        list[dict[str, dict[str, str]]], TestClient(app).get("/events").json()
    )
    assert [e["TestEvent"]["event_id"] for e in returned] == [
        str(e.event_id) for e in events[-RECENT_EVENTS:]
    ]
