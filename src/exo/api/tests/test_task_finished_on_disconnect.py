# pyright: reportAny=false
"""Tests that generation streams tell the master their command finished even
when the client disconnects (cancelling the stream) mid-generation."""

from collections.abc import Awaitable, Callable

import anyio
import pytest
from fastapi import Request

from exo.api.main import API
from exo.shared.types.commands import (
    ForwarderCommand,
    TaskCancelled,
    TaskFinished,
    TextGeneration,
)
from exo.shared.types.common import CommandId, ModelId, SystemId
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.utils.channels import Receiver, channel


def _make_api() -> tuple[API, Receiver[ForwarderCommand]]:
    api = object.__new__(API)
    api.paused = False
    api._system_id = SystemId()  # pyright: ignore[reportPrivateUsage]
    api._text_generation_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._image_generation_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._cancelled_command_ids = set()  # pyright: ignore[reportPrivateUsage]
    api.command_sender, command_receiver = channel[ForwarderCommand]()
    return api, command_receiver


async def _stream_text(api: API, command_id: CommandId) -> None:
    text_generation = TextGeneration(
        command_id=command_id,
        task_params=TextGenerationTaskParams(
            model=ModelId("test-model"),
            input=[InputMessage(role="user", content=InputMessageContent("hello"))],
        ),
    )
    async for _ in api._token_chunk_stream(text_generation):  # pyright: ignore[reportPrivateUsage]
        pass


async def _stream_images(api: API, command_id: CommandId) -> None:
    async for _ in api._generate_image_stream(  # pyright: ignore[reportPrivateUsage]
        request=Request({"type": "http"}),
        command_id=command_id,
        num_images=1,
        response_format="b64_json",
    ):
        pass


async def _collect_images(api: API, command_id: CommandId) -> None:
    await api._collect_image_chunks(  # pyright: ignore[reportPrivateUsage]
        request=None,
        command_id=command_id,
        num_images=1,
        response_format="b64_json",
    )


def _has_queue(api: API, command_id: CommandId) -> bool:
    return (
        command_id in api._text_generation_queues  # pyright: ignore[reportPrivateUsage]
        or command_id in api._image_generation_queues  # pyright: ignore[reportPrivateUsage]
    )


@pytest.mark.parametrize(
    "consume_stream",
    [_stream_text, _stream_images, _collect_images],
    ids=["text_stream", "image_stream", "image_collect"],
)
async def test_task_finished_sent_when_stream_is_cancelled(
    consume_stream: Callable[[API, CommandId], Awaitable[None]],
) -> None:
    api, command_receiver = _make_api()
    command_id = CommandId("command")

    with anyio.fail_after(5):
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(consume_stream, api, command_id)
            while not _has_queue(api, command_id):
                await anyio.sleep(0)
            # The client disconnects while the stream is waiting for chunks
            task_group.cancel_scope.cancel()

    sent = [forwarded.command for forwarded in command_receiver.collect()]
    assert [type(command) for command in sent] == [TaskCancelled, TaskFinished]
    task_finished = sent[1]
    assert isinstance(task_finished, TaskFinished)
    assert task_finished.finished_command_id == command_id
    assert not _has_queue(api, command_id)
