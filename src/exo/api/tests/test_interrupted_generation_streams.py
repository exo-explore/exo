# pyright: reportAny=false
"""Tests that generation requests end with an explicit error, rather than
hanging or silently truncating, when their stream is closed before the
generation finished (instance deleted, node dropped, or master changed)."""

import json
from unittest.mock import AsyncMock

import anyio
import pytest
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse

from exo.api.main import API
from exo.api.recent_events import RecentEvents
from exo.api.types import (
    ChatCompletionMessage,
    ChatCompletionRequest,
    ImageGenerationTaskParams,
)
from exo.shared.types.chunks import TokenChunk
from exo.shared.types.commands import ForwarderCommand
from exo.shared.types.common import CommandId, ModelId, SystemId
from exo.shared.types.events import IndexedEvent
from exo.shared.types.state import State
from exo.shared.types.tasks import ImageGeneration, Task, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId
from exo.utils.channels import Receiver, channel
from exo.utils.task_group import TaskGroup

_MODEL = ModelId("test-model")
_INSTANCE_ID = InstanceId("instance")
_INTERRUPTED_MESSAGE = (
    "The model instance serving this request stopped before it finished"
)
_DONE_EVENT = "data: [DONE]\n\n"


def _make_api() -> tuple[API, Receiver[ForwarderCommand]]:
    api = object.__new__(API)
    api.state = State()
    api.paused = False
    api._system_id = SystemId()  # pyright: ignore[reportPrivateUsage]
    api._text_generation_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._image_generation_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._cancelled_command_ids = set()  # pyright: ignore[reportPrivateUsage]
    api._send = AsyncMock()  # pyright: ignore[reportPrivateUsage]
    api._validate_model_has_instance = AsyncMock(return_value=_MODEL)  # pyright: ignore[reportPrivateUsage]
    api.command_sender, command_receiver = channel[ForwarderCommand]()
    return api, command_receiver


async def _start_chat_completion(api: API, *, stream: bool) -> StreamingResponse:
    response = await api.chat_completions(
        ChatCompletionRequest(
            model=_MODEL,
            messages=[ChatCompletionMessage(role="user", content="hello")],
            stream=stream,
        )
    )
    assert isinstance(response, StreamingResponse)
    return response


async def _read_body(response: StreamingResponse, events: list[str]) -> None:
    async for event in response.body_iterator:
        assert isinstance(event, str)
        events.append(event)


async def _wait_for_queue(api: API) -> CommandId:
    while not (
        api._text_generation_queues  # pyright: ignore[reportPrivateUsage]
        or api._image_generation_queues  # pyright: ignore[reportPrivateUsage]
    ):
        await anyio.sleep(0)
    return next(
        iter(
            api._text_generation_queues  # pyright: ignore[reportPrivateUsage]
            or api._image_generation_queues  # pyright: ignore[reportPrivateUsage]
        )
    )


async def _send_token(api: API, command_id: CommandId) -> None:
    await api._text_generation_queues[command_id].send(  # pyright: ignore[reportPrivateUsage]
        TokenChunk(model=_MODEL, text="Hel", token_id=1, usage=None)
    )


def _delete_instance(api: API, task: Task) -> None:
    """Close the streams of every command on the instance, as the API does
    when it applies an InstanceDeleted event."""
    api.state = State(tasks={task.task_id: task})
    api._close_streams_for_instance(_INSTANCE_ID)  # pyright: ignore[reportPrivateUsage]


def _text_task(command_id: CommandId) -> TextGeneration:
    return TextGeneration(
        instance_id=_INSTANCE_ID,
        command_id=command_id,
        task_params=TextGenerationTaskParams(
            model=_MODEL,
            input=[InputMessage(role="user", content=InputMessageContent("hello"))],
        ),
    )


def _image_task(command_id: CommandId) -> ImageGeneration:
    return ImageGeneration(
        instance_id=_INSTANCE_ID,
        command_id=command_id,
        task_params=ImageGenerationTaskParams(prompt="a cat", model=_MODEL),
    )


def _error_message(sse_event: str) -> str:
    payload: dict[str, dict[str, str]] = json.loads(sse_event.removeprefix("data: "))
    return payload["error"]["message"]


async def test_chat_stream_reports_error_when_instance_is_deleted() -> None:
    api, _commands = _make_api()
    response = await _start_chat_completion(api, stream=True)
    events: list[str] = []

    with anyio.fail_after(5):
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(_read_body, response, events)
            command_id = await _wait_for_queue(api)
            await _send_token(api, command_id)
            _delete_instance(api, _text_task(command_id))

    assert events[-1:] == [_DONE_EVENT]
    assert _error_message(events[-2]) == _INTERRUPTED_MESSAGE


async def test_chat_completion_fails_when_instance_is_deleted() -> None:
    """A non-streaming client must not receive the partial output as if the
    generation had completed."""
    api, _commands = _make_api()
    response = await _start_chat_completion(api, stream=False)

    async def read_body_expecting_error() -> None:
        with pytest.raises(ValueError, match=_INTERRUPTED_MESSAGE):
            await _read_body(response, [])

    with anyio.fail_after(5):
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(read_body_expecting_error)
            command_id = await _wait_for_queue(api)
            await _send_token(api, command_id)
            _delete_instance(api, _text_task(command_id))


async def test_image_stream_reports_error_when_instance_is_deleted() -> None:
    api, _commands = _make_api()
    command_id = CommandId("image-command")
    events: list[str] = []

    async def read_image_stream() -> None:
        async for event in api._generate_image_stream(  # pyright: ignore[reportPrivateUsage]
            request=Request({"type": "http"}),
            command_id=command_id,
            num_images=1,
            response_format="b64_json",
        ):
            events.append(event)

    with anyio.fail_after(5):
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(read_image_stream)
            await _wait_for_queue(api)
            _delete_instance(api, _image_task(command_id))

    assert events[-1:] == [_DONE_EVENT]
    assert _error_message(events[-2]) == _INTERRUPTED_MESSAGE


async def test_image_collect_fails_when_instance_is_deleted() -> None:
    api, _commands = _make_api()
    command_id = CommandId("image-command")

    async def collect_images_expecting_error() -> None:
        with pytest.raises(HTTPException) as raised:
            await api._collect_image_chunks(  # pyright: ignore[reportPrivateUsage]
                request=None,
                command_id=command_id,
                num_images=1,
                response_format="b64_json",
            )
        assert raised.value.detail == _INTERRUPTED_MESSAGE

    with anyio.fail_after(5):
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(collect_images_expecting_error)
            await _wait_for_queue(api)
            _delete_instance(api, _image_task(command_id))


async def test_reset_ends_in_flight_streams_with_error() -> None:
    """A master change resets the API; requests in flight must end with an
    error instead of hanging forever."""
    api, _commands = _make_api()
    api._recent_events = RecentEvents()  # pyright: ignore[reportPrivateUsage]
    api.paused_ev = anyio.Event()
    api.last_completed_election = 0
    _old_events, api.event_receiver = channel[IndexedEvent]()
    api._tg = TaskGroup()  # pyright: ignore[reportPrivateUsage]
    response = await _start_chat_completion(api, stream=True)
    events: list[str] = []

    with anyio.fail_after(5):
        async with api._tg as task_group:  # pyright: ignore[reportPrivateUsage]
            task_group.start_soon(_read_body, response, events)
            await _wait_for_queue(api)
            new_events, new_event_receiver = channel[IndexedEvent]()
            api.reset(result_clock=1, event_receiver=new_event_receiver)
            new_events.close()

    assert events[-1:] == [_DONE_EVENT]
    assert _error_message(events[-2]) == _INTERRUPTED_MESSAGE


async def test_cancelled_stream_ends_without_error() -> None:
    """Cancelling through /v1/cancel also closes the stream, but the user asked
    for it, so it must not be reported as a failure."""
    api, _commands = _make_api()
    response = await _start_chat_completion(api, stream=True)
    events: list[str] = []

    with anyio.fail_after(5):
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(_read_body, response, events)
            command_id = await _wait_for_queue(api)
            await api.cancel_command(command_id)

    assert not any('"error"' in event for event in events)
