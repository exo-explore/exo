"""A request for a model with no running instance ends with an error instead of being dropped.

The API checks that a model is running before it sends a request, but the instance can be
deleted before the master handles it (a node timing out, say). The master used to log the
error and drop the request, leaving its stream open forever.
"""

import anyio
import pytest

from exo.api.types import ImageGenerationTaskParams
from exo.master.main import Master
from exo.routing.router import get_node_zid
from exo.shared.models.model_cards import ModelId
from exo.shared.types.chunks import ErrorChunk
from exo.shared.types.commands import (
    Command,
    ForwarderCommand,
    ForwarderDownloadCommand,
    ImageGeneration,
    TextGeneration,
)
from exo.shared.types.common import SessionId, SystemId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    GlobalForwarderEvent,
    LocalForwarderEvent,
)
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.utils.channels import channel

MODEL = ModelId("mlx-community/not-running")


def chat_request() -> TextGeneration:
    return TextGeneration(
        task_params=TextGenerationTaskParams(
            model=MODEL,
            input=[InputMessage(role="user", content=InputMessageContent("hi"))],
        )
    )


def image_request() -> ImageGeneration:
    return ImageGeneration(
        task_params=ImageGenerationTaskParams(prompt="a cat", model=MODEL)
    )


@pytest.mark.parametrize("request_command", [chat_request(), image_request()])
async def test_a_request_for_a_model_with_no_instance_ends_with_an_error(
    request_command: Command,
) -> None:
    node_id = get_node_zid()
    command_sender, commands = channel[ForwarderCommand]()
    event_sender, events = channel[Event]()
    master = Master(
        node_id,
        SessionId(master_node_id=node_id, election_clock=0),
        command_receiver=commands,
        event_sender=event_sender,
        local_event_receiver=channel[LocalForwarderEvent]()[1],
        global_event_sender=channel[GlobalForwarderEvent]()[0],
        download_command_sender=channel[ForwarderDownloadCommand]()[0],
    )

    async with anyio.create_task_group() as tg:
        tg.start_soon(master._command_processor)  # pyright: ignore[reportPrivateUsage]
        await command_sender.send(
            ForwarderCommand(origin=SystemId(), command=request_command)
        )
        with anyio.fail_after(5):
            event = await events.receive()
        tg.cancel_scope.cancel()

        assert isinstance(event, ChunkGenerated)
        assert event.command_id == request_command.command_id
        assert event.chunk == ErrorChunk(
            model=MODEL, error_message=f"No instance found for model {MODEL}"
        )
