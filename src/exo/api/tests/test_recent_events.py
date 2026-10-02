"""The API remembers only the most recent events, so its history can't grow without bound."""

from typing import cast

from fastapi import FastAPI
from fastapi.testclient import TestClient

from exo.api.main import API
from exo.api.recent_events import RECENT_EVENTS, RecentEvents
from exo.shared.types.chunks import InputImageChunk
from exo.shared.types.common import CommandId, ModelId
from exo.shared.types.events import IndexedEvent, InputChunkReceived, TestEvent
from exo.shared.types.state import State
from exo.utils.channels import channel


async def test_events_endpoint_returns_the_most_recent_events() -> None:
    events = [TestEvent() for _ in range(RECENT_EVENTS + 5)]
    send, recv = channel[IndexedEvent]()
    api = object.__new__(API)
    api.state = State()
    api.event_receiver = recv
    api._recent_events = RecentEvents()  # pyright: ignore[reportPrivateUsage]
    api._text_generation_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._image_generation_queues = {}  # pyright: ignore[reportPrivateUsage]
    for idx, event in enumerate(events):
        await send.send(IndexedEvent(idx=idx, event=event))
    send.close()

    await api._apply_state()  # pyright: ignore[reportPrivateUsage]

    app = FastAPI()
    app.get("/events")(api.get_events)
    returned = cast(
        list[dict[str, dict[str, str]]], TestClient(app).get("/events").json()
    )
    assert [e["TestEvent"]["event_id"] for e in returned] == [
        str(e.event_id) for e in events[-RECENT_EVENTS:]
    ]


def test_image_data_beyond_its_budget_is_dropped_first() -> None:
    def image_chunk() -> InputChunkReceived:
        command_id = CommandId()
        return InputChunkReceived(
            command_id=command_id,
            chunk=InputImageChunk(
                model=ModelId("test-model"),
                command_id=command_id,
                data="x" * 60,
                chunk_index=0,
                total_chunks=1,
            ),
        )

    recent = RecentEvents(max_events=10, max_image_bytes=100)
    first, second, plain = image_chunk(), image_chunk(), TestEvent()
    for event in (first, second, plain):
        recent.append(event)

    assert list(recent) == [second, plain]
