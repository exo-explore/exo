# pyright: reportUnusedFunction=false, reportAny=false
"""Tests that streaming queues are reconciled to live state.

Earlier the API reacted to `InstanceDeleted` events; now it reconciles its
queues against `state.tasks` × `state.instances`. The tests assert the same
end-state via the reconciliation pass instead of the old direct method.
"""

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
from exo.shared.types.worker.instances import InstanceId, MlxRingInstance
from exo.shared.types.worker.runners import ShardAssignments


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


def _live_command_ids(api: API) -> set[CommandId]:
    return {
        task.command_id
        for task in api.state.tasks.values()
        if isinstance(task, (TextGeneration, ImageGeneration))
        and task.instance_id in api.state.instances
    }


def test_close_streams_when_instance_missing_from_state() -> None:
    """A command whose instance is no longer in state.instances has its stream closed."""
    instance_id = InstanceId("inst-1")
    command_id = CommandId("cmd-1")
    task = _make_text_gen_task(instance_id, command_id)

    # Instance is gone (deleted); task entry still references it.
    state = State(tasks={task.task_id: task}, instances={})
    api = _make_api_with_state(state)

    sender = MagicMock()
    api._text_generation_queues[command_id] = sender  # pyright: ignore[reportPrivateUsage]

    api._close_streams_not_in(_live_command_ids(api))  # pyright: ignore[reportPrivateUsage]

    sender.close.assert_called_once()
    assert command_id not in api._text_generation_queues  # pyright: ignore[reportPrivateUsage]


def test_keeps_streams_for_live_commands() -> None:
    """Streams for tasks whose instance still exists are left open."""
    live_id = InstanceId("inst-live")
    live_cmd = CommandId("cmd-live")
    live_task = _make_text_gen_task(live_id, live_cmd)

    # Reconciliation only checks key membership in state.instances. We
    # construct a minimal MlxRingInstance so Pydantic's State validator is
    # happy; the contents don't matter to the test.
    placeholder_instance = MlxRingInstance(
        instance_id=live_id,
        shard_assignments=ShardAssignments(
            model_id=ModelId("test-model"), node_to_runner={}, runner_to_shard={}
        ),
        hosts_by_node={},
        ephemeral_port=0,
    )
    state = State(
        tasks={live_task.task_id: live_task},
        instances={live_id: placeholder_instance},
    )
    api = _make_api_with_state(state)

    sender = MagicMock()
    api._text_generation_queues[live_cmd] = sender  # pyright: ignore[reportPrivateUsage]

    api._close_streams_not_in(_live_command_ids(api))  # pyright: ignore[reportPrivateUsage]

    sender.close.assert_not_called()
    assert live_cmd in api._text_generation_queues  # pyright: ignore[reportPrivateUsage]


def test_close_image_generation_stream_when_instance_missing() -> None:
    """The same reconciliation path closes image generation streams."""
    instance_id = InstanceId("inst-img")
    command_id = CommandId("cmd-img")
    task = ImageGeneration(
        instance_id=instance_id,
        command_id=command_id,
        task_params=ImageGenerationTaskParams(prompt="a cat", model="test-model"),
    )

    state = State(tasks={task.task_id: task}, instances={})
    api = _make_api_with_state(state)

    sender = MagicMock()
    api._image_generation_queues[command_id] = sender  # pyright: ignore[reportPrivateUsage]

    api._close_streams_not_in(_live_command_ids(api))  # pyright: ignore[reportPrivateUsage]

    sender.close.assert_called_once()
    assert command_id not in api._image_generation_queues  # pyright: ignore[reportPrivateUsage]
