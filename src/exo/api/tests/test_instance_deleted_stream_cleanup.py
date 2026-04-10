# pyright: reportUnusedFunction=false, reportAny=false
"""Tests that InstanceDeleted events close active generation streams."""

from unittest.mock import MagicMock

from exo.api.main import API
from exo.api.types import ImageGenerationTaskParams
from exo.shared.types.common import CommandId, ModelId
from exo.shared.types.state import State
from exo.shared.types.tasks import ImageGeneration, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId


def _make_api_with_state(state: State) -> API:
    """Create a minimal API instance with pre-set state."""
    api = object.__new__(API)
    api.state = state
    api._text_generation_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._image_generation_queues = {}  # pyright: ignore[reportPrivateUsage]
    return api


def _make_text_gen_task(
    instance_id: InstanceId, command_id: CommandId
) -> TextGeneration:
    return TextGeneration(
        instance_id=instance_id,
        command_id=command_id,
        task_params=TextGenerationTaskParams(
            model=ModelId("test-model"),
            input=[InputMessage(role="user", content=InputMessageContent("hello"))],
        ),
    )


def test_close_streams_for_deleted_instance() -> None:
    """Deleting an instance closes the text generation sender for commands on that instance."""
    instance_id = InstanceId("inst-1")
    command_id = CommandId("cmd-1")
    task = _make_text_gen_task(instance_id, command_id)

    state = State(tasks={task.task_id: task})
    api = _make_api_with_state(state)

    sender = MagicMock()
    api._text_generation_queues[command_id] = sender  # pyright: ignore[reportPrivateUsage]

    api._close_streams_for_instance(instance_id)  # pyright: ignore[reportPrivateUsage]

    sender.close.assert_called_once()
    assert command_id not in api._text_generation_queues  # pyright: ignore[reportPrivateUsage]


def test_close_streams_ignores_unrelated_instances() -> None:
    """Deleting an instance does NOT close streams for commands on other instances."""
    target_id = InstanceId("inst-delete")
    other_id = InstanceId("inst-keep")
    other_cmd = CommandId("cmd-keep")
    other_task = _make_text_gen_task(other_id, other_cmd)

    state = State(tasks={other_task.task_id: other_task})
    api = _make_api_with_state(state)

    sender = MagicMock()
    api._text_generation_queues[other_cmd] = sender  # pyright: ignore[reportPrivateUsage]

    api._close_streams_for_instance(target_id)  # pyright: ignore[reportPrivateUsage]

    sender.close.assert_not_called()
    assert other_cmd in api._text_generation_queues  # pyright: ignore[reportPrivateUsage]


def test_close_streams_for_deleted_instance_image_generation() -> None:
    """Deleting an instance closes the image generation sender for commands on that instance."""
    instance_id = InstanceId("inst-img")
    command_id = CommandId("cmd-img")
    task = ImageGeneration(
        instance_id=instance_id,
        command_id=command_id,
        task_params=ImageGenerationTaskParams(prompt="a cat", model="test-model"),
    )

    state = State(tasks={task.task_id: task})
    api = _make_api_with_state(state)

    sender = MagicMock()
    api._image_generation_queues[command_id] = sender  # pyright: ignore[reportPrivateUsage]

    api._close_streams_for_instance(instance_id)  # pyright: ignore[reportPrivateUsage]

    sender.close.assert_called_once()
    assert command_id not in api._image_generation_queues  # pyright: ignore[reportPrivateUsage]
