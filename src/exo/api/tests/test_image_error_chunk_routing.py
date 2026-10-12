# pyright: reportAny=false
"""Tests that ErrorChunks for image commands reach the image stream consumer."""

import json
from unittest.mock import AsyncMock

import anyio
from fastapi import Request

from exo.api.main import API
from exo.api.recent_events import RecentEvents
from exo.shared.types.chunks import ErrorChunk
from exo.shared.types.commands import ForwarderCommand
from exo.shared.types.common import CommandId, ModelId, SystemId
from exo.shared.types.events import ChunkGenerated, IndexedEvent
from exo.shared.types.state import State
from exo.utils.channels import channel


def _make_api() -> API:
    api = object.__new__(API)
    api.state = State()
    api._recent_events = RecentEvents()  # pyright: ignore[reportPrivateUsage]
    api._text_generation_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._image_generation_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._cancelled_command_ids = set()  # pyright: ignore[reportPrivateUsage]
    api._send = AsyncMock()  # pyright: ignore[reportPrivateUsage]
    api._system_id = SystemId()  # pyright: ignore[reportPrivateUsage]
    api.command_sender, _ = channel[ForwarderCommand]()
    return api


async def test_image_error_chunk_reaches_image_stream() -> None:
    """A runner error for an image command is streamed to the client instead of
    crashing the API's event loop."""
    api = _make_api()
    event_sender, api.event_receiver = channel[IndexedEvent]()
    command_id = CommandId("image-command")
    error_chunk = ErrorChunk(
        model=ModelId("test-image-model"),
        error_message="Runner shutdown before completing command",
    )
    streamed_events: list[str] = []

    async def consume_image_stream() -> None:
        async for event in api._generate_image_stream(  # pyright: ignore[reportPrivateUsage]
            request=Request({"type": "http"}),
            command_id=command_id,
            num_images=1,
            response_format="b64_json",
        ):
            streamed_events.append(event)

    with anyio.fail_after(5):
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(consume_image_stream)
            while command_id not in api._image_generation_queues:  # pyright: ignore[reportPrivateUsage]
                await anyio.sleep(0)

            await event_sender.send(
                IndexedEvent(
                    idx=0,
                    event=ChunkGenerated(command_id=command_id, chunk=error_chunk),
                )
            )
            event_sender.close()
            await api._apply_state()  # pyright: ignore[reportPrivateUsage]

    assert streamed_events[-1] == "data: [DONE]\n\n"
    error_payload: dict[str, dict[str, str]] = json.loads(
        streamed_events[-2].removeprefix("data: ")
    )
    assert error_payload["error"]["message"] == error_chunk.error_message
