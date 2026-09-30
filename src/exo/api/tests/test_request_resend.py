"""A chat request whose command never reaches the master is sent again, then ends with an error."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import anyio
import pytest

import exo.api.main as api_main
from exo.api.main import API
from exo.shared.models.model_cards import ModelId
from exo.shared.types.chunks import (
    ErrorChunk,
    InputImageChunk,
    PrefillProgressChunk,
    TokenChunk,
    ToolCallChunk,
)
from exo.shared.types.commands import (
    Command,
    ForwarderCommand,
    SendInputChunk,
    TextGeneration,
)
from exo.shared.types.common import SystemId
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.task_group import TaskGroup

MODEL = ModelId("test-model")
type Chunk = TokenChunk | ErrorChunk | ToolCallChunk | PrefillProgressChunk


@pytest.fixture(autouse=True)
def quick_resends(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_main, "REQUEST_RESEND_WAITS", (0.05, 0.05))


def chat_request() -> TextGeneration:
    return TextGeneration(
        task_params=TextGenerationTaskParams(
            model=MODEL,
            input=[InputMessage(role="user", content=InputMessageContent("hi"))],
        )
    )


@asynccontextmanager
async def running_api() -> AsyncIterator[tuple[API, Receiver[ForwarderCommand]]]:
    api = object.__new__(API)
    api.paused = False
    api._system_id = SystemId()  # pyright: ignore[reportPrivateUsage]
    api.command_sender, commands = channel[ForwarderCommand]()
    api._text_generation_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._tg = TaskGroup()  # pyright: ignore[reportPrivateUsage]
    async with api._tg:  # pyright: ignore[reportPrivateUsage]
        yield api, commands
        api._tg.cancel_tasks()  # pyright: ignore[reportPrivateUsage]


def open_stream(api: API, request: TextGeneration) -> Receiver[Chunk]:
    send: Sender[Chunk]
    send, receive = channel[Chunk]()
    api._text_generation_queues[request.command_id] = send  # pyright: ignore[reportPrivateUsage]
    return receive


def sent(commands: Receiver[ForwarderCommand]) -> list[Command]:
    return [c.command for c in commands.collect()]


async def test_a_request_never_accepted_is_sent_again_then_ends_with_an_error() -> None:
    request = chat_request()
    async with running_api() as (api, commands):
        stream = open_stream(api, request)
        await api._send(request)  # pyright: ignore[reportPrivateUsage]
        with anyio.fail_after(5):
            chunk = await stream.receive()

    assert sent(commands) == [request, request, request]
    assert isinstance(chunk, ErrorChunk)
    assert "didn't accept" in chunk.error_message


async def test_an_accepted_request_is_not_sent_again() -> None:
    request = chat_request()
    async with running_api() as (api, commands):
        open_stream(api, request)
        await api._send(request)  # pyright: ignore[reportPrivateUsage]
        api._mark_accepted(request.command_id)  # pyright: ignore[reportPrivateUsage]
        await anyio.sleep(0.2)

    assert sent(commands) == [request]


async def test_a_request_that_has_ended_is_not_sent_again() -> None:
    request = chat_request()
    async with running_api() as (api, commands):
        await api._send(request)  # pyright: ignore[reportPrivateUsage]
        await anyio.sleep(0.2)

    assert sent(commands) == [request]


async def test_image_chunks_are_sent_again_with_the_request() -> None:
    request = chat_request()
    chunk = SendInputChunk(
        chunk=InputImageChunk(
            model=MODEL,
            command_id=request.command_id,
            data="aGk=",
            chunk_index=0,
            total_chunks=1,
        )
    )
    async with running_api() as (api, commands):
        open_stream(api, request)
        await api._send(chunk)  # pyright: ignore[reportPrivateUsage]
        await api._send(request)  # pyright: ignore[reportPrivateUsage]
        await anyio.sleep(0.07)
        api._mark_accepted(request.command_id)  # pyright: ignore[reportPrivateUsage]
        await anyio.sleep(0.1)

    assert sent(commands) == [chunk, request, chunk, request]
