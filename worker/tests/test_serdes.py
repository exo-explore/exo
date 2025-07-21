from typing import Callable, TypeVar

from pydantic import BaseModel, TypeAdapter

from shared.types.tasks.common import Task
from shared.types.worker.commands_runner import (
    ChatTaskMessage,
    RunnerMessageTypeAdapter,
    SetupMessage,
)
from shared.types.worker.mlx import Host
from shared.types.worker.shards import PipelineShardMetadata

T = TypeVar("T", bound=BaseModel)


def assert_equal_serdes(obj: T, typeadapter: TypeAdapter[T]):
    encoded: bytes = obj.model_dump_json().encode("utf-8") + b"\n"
    decoded: T = typeadapter.validate_json(encoded)

    assert decoded == obj, (
        f"Decoded: {decoded} != \nOriginal: {obj}. \n binary encoded: {encoded}"
    )


def test_supervisor_setup_message_serdes(
    pipeline_shard_meta: Callable[..., PipelineShardMetadata],
    hosts: Callable[..., list[Host]],
):
    setup_message = SetupMessage(
        model_shard_meta=pipeline_shard_meta(1, 0),
        hosts=hosts(1),
    )
    assert_equal_serdes(setup_message, RunnerMessageTypeAdapter)


def test_supervisor_task_message_serdes(
    chat_completion_task: Task,
):
    task_message = ChatTaskMessage(
        task_data=chat_completion_task.task_params,
    )
    assert_equal_serdes(task_message, RunnerMessageTypeAdapter)
