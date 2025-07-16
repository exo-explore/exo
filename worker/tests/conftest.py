import uuid
from pathlib import Path
from typing import Callable, cast

import pytest
from openai.types.chat import ChatCompletionUserMessageParam
from openai.types.chat.completion_create_params import (
    CompletionCreateParamsNonStreaming,
    CompletionCreateParamsStreaming,
)
from pydantic import TypeAdapter

from shared.types.models.common import ModelId
from shared.types.tasks.common import (
    ChatCompletionStreamingTask,
    Task,
    TaskArtifact,
    TaskId,
    TaskState,
    TaskStatusOtherType,
    TaskStatusType,
    TaskType,
)
from shared.types.worker.common import InstanceId
from shared.types.worker.mlx import Host
from shared.types.worker.shards import PipelineShardMetadata

CompletionCreateParamsStreamingAdapter = TypeAdapter(CompletionCreateParamsStreaming)
CompletionCreateParamsNonStreamingAdapter = TypeAdapter(
    CompletionCreateParamsNonStreaming
)


# Concrete TaskArtifact implementation for pending streaming tasks
class PendingStreamingTaskArtifact(
    TaskArtifact[TaskType.ChatCompletionStreaming, TaskStatusOtherType.Pending]
):
    pass


@pytest.fixture
def pipeline_shard_meta():
    def _pipeline_shard_meta(
        num_nodes: int = 1, device_rank: int = 0
    ) -> PipelineShardMetadata:
        total_layers = 16
        layers_per_node = total_layers // num_nodes
        start_layer = device_rank * layers_per_node
        end_layer = (
            start_layer + layers_per_node
            if device_rank < num_nodes - 1
            else total_layers
        )

        return PipelineShardMetadata(
            device_rank=device_rank,
            model_id=ModelId(uuid=uuid.uuid4()),
            model_path=Path(
                "~/.exo/models/mlx-community--Llama-3.2-1B-Instruct-4bit/"
            ).expanduser(),
            start_layer=start_layer,
            end_layer=end_layer,
            world_size=num_nodes,
        )

    return _pipeline_shard_meta


@pytest.fixture
def hosts():
    def _hosts(count: int, offset: int = 0) -> list[Host]:
        return [
            Host(
                host="127.0.0.1",
                port=5000 + offset + i,
            )
            for i in range(count)
        ]

    return _hosts


@pytest.fixture
def hosts_one(hosts: Callable[[int], list[Host]]):
    return hosts(1)


@pytest.fixture
def hosts_two(hosts: Callable[[int], list[Host]]):
    return hosts(2)


@pytest.fixture
def user_message():
    """Override this fixture in tests to customize the message"""
    return "Hello, how are you?"


@pytest.fixture
def chat_completion_params(user_message: str):
    """Creates ChatCompletionParams with the given message"""
    return CompletionCreateParamsStreaming(
        model="gpt-4",
        messages=[ChatCompletionUserMessageParam(role="user", content=user_message)],
        stream=True,
    )


@pytest.fixture
def chat_completion_streaming_task_data(
    chat_completion_params: CompletionCreateParamsStreaming,
):
    """Creates ChatCompletionStreamingTask from params"""
    return ChatCompletionStreamingTask(task_data=chat_completion_params)


@pytest.fixture
def streaming_task(
    chat_completion_streaming_task_data: CompletionCreateParamsStreaming,
) -> Task[TaskType, TaskStatusType]:
    """Creates the final Task object"""
    task = Task(
        task_id=TaskId(),
        task_type=TaskType.ChatCompletionStreaming,
        task_params=ChatCompletionStreamingTask(
            task_data=chat_completion_streaming_task_data
        ),
        task_state=TaskState(
            task_status=TaskStatusOtherType.Pending,
            task_artifact=PendingStreamingTaskArtifact(),
        ),
        on_instance=InstanceId(),
    )
    return cast(Task[TaskType, TaskStatusType], task)
