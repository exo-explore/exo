"""Every request sends its own images, even ones an earlier request already sent."""

from exo.api.main import API
from exo.shared.models.model_cards import ModelId
from exo.shared.types.commands import ForwarderCommand, SendInputChunk, TextGeneration
from exo.shared.types.common import SystemId
from exo.shared.types.text_generation import (
    Base64Image,
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.utils.channels import channel
from exo.utils.task_group import TaskGroup

IMAGE = Base64Image("aGVsbG8gd29ybGQ=")


async def test_an_image_is_sent_again_with_every_request_that_uses_it() -> None:
    api = object.__new__(API)
    api.paused = False
    api._system_id = SystemId()  # pyright: ignore[reportPrivateUsage]
    api.command_sender, commands = channel[ForwarderCommand]()
    api._tg = TaskGroup()  # pyright: ignore[reportPrivateUsage]
    params = TextGenerationTaskParams(
        model=ModelId("test-org/vision-model"),
        input=[InputMessage(role="user", content=InputMessageContent("What's this?"))],
        images=[IMAGE],
    )

    for _ in range(2):
        await api._send_text_generation_with_images(params)  # pyright: ignore[reportPrivateUsage]

    sent = [c.command for c in commands.collect()]
    assert [type(c) for c in sent] == [
        SendInputChunk,
        TextGeneration,
        SendInputChunk,
        TextGeneration,
    ]
