# pyright: reportUnusedFunction=false, reportAny=false
"""Tests that streaming queues reconcile against durable State."""

from unittest.mock import MagicMock

from exo.api.main import API
from exo.api.types import ImageGenerationTaskParams
from exo.shared.types.common import CommandId, ModelId, NodeId
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
    api = object.__new__(API)
    api.state = state
    api._text_generation_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._image_generation_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._observed_generation_commands = set()  # pyright: ignore[reportPrivateUsage]
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


def _make_instance(instance_id: InstanceId) -> MlxRingInstance:
    return MlxRingInstance(
        instance_id=instance_id,
        shard_assignments=ShardAssignments(
            model_id=ModelId("test-model"),
            node_to_runner={},
            runner_to_shard={},
        ),
        hosts_by_node={NodeId("node-1"): []},
        ephemeral_port=1,
    )


def test_reconcile_closes_stream_when_task_instance_is_missing() -> None:
    instance_id = InstanceId("inst-1")
    command_id = CommandId("cmd-1")
    task = _make_text_gen_task(instance_id, command_id)
    api = _make_api_with_state(State(tasks={task.task_id: task}, instances={}))

    sender = MagicMock()
    api._text_generation_queues[command_id] = sender  # pyright: ignore[reportPrivateUsage]

    api._reconcile_streams_once()  # pyright: ignore[reportPrivateUsage]

    sender.close.assert_called_once()
    assert command_id not in api._text_generation_queues  # pyright: ignore[reportPrivateUsage]


def test_reconcile_keeps_stream_for_live_task_instance() -> None:
    instance_id = InstanceId("inst-live")
    command_id = CommandId("cmd-live")
    task = _make_text_gen_task(instance_id, command_id)
    api = _make_api_with_state(
        State(
            tasks={task.task_id: task},
            instances={instance_id: _make_instance(instance_id)},
        )
    )

    sender = MagicMock()
    api._text_generation_queues[command_id] = sender  # pyright: ignore[reportPrivateUsage]

    api._reconcile_streams_once()  # pyright: ignore[reportPrivateUsage]

    sender.close.assert_not_called()
    assert command_id in api._text_generation_queues  # pyright: ignore[reportPrivateUsage]


def test_reconcile_does_not_close_command_before_state_observes_it() -> None:
    command_id = CommandId("cmd-not-created-yet")
    api = _make_api_with_state(State())

    sender = MagicMock()
    api._text_generation_queues[command_id] = sender  # pyright: ignore[reportPrivateUsage]

    api._reconcile_streams_once()  # pyright: ignore[reportPrivateUsage]

    sender.close.assert_not_called()
    assert command_id in api._text_generation_queues  # pyright: ignore[reportPrivateUsage]


def test_reconcile_closes_stream_after_observed_task_leaves_state() -> None:
    instance_id = InstanceId("inst-live")
    command_id = CommandId("cmd-deleted")
    task = _make_text_gen_task(instance_id, command_id)
    api = _make_api_with_state(
        State(
            tasks={task.task_id: task},
            instances={instance_id: _make_instance(instance_id)},
        )
    )

    sender = MagicMock()
    api._text_generation_queues[command_id] = sender  # pyright: ignore[reportPrivateUsage]
    api._reconcile_streams_once()  # pyright: ignore[reportPrivateUsage]

    api.state = State(instances={instance_id: _make_instance(instance_id)})
    api._reconcile_streams_once()  # pyright: ignore[reportPrivateUsage]

    sender.close.assert_called_once()
    assert command_id not in api._text_generation_queues  # pyright: ignore[reportPrivateUsage]


def test_reconcile_closes_image_stream_when_task_instance_is_missing() -> None:
    instance_id = InstanceId("inst-img")
    command_id = CommandId("cmd-img")
    task = ImageGeneration(
        instance_id=instance_id,
        command_id=command_id,
        task_params=ImageGenerationTaskParams(prompt="a cat", model="test-model"),
    )
    api = _make_api_with_state(State(tasks={task.task_id: task}, instances={}))

    sender = MagicMock()
    api._image_generation_queues[command_id] = sender  # pyright: ignore[reportPrivateUsage]

    api._reconcile_streams_once()  # pyright: ignore[reportPrivateUsage]

    sender.close.assert_called_once()
    assert command_id not in api._image_generation_queues  # pyright: ignore[reportPrivateUsage]
