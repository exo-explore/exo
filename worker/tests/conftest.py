from ipaddress import IPv4Address
from logging import Logger, getLogger
from typing import Callable, Optional

import pytest

from shared.models.model_meta import get_model_meta
from shared.types.api import ChatCompletionMessage, ChatCompletionTaskParams
from shared.types.common import Host, NodeId
from shared.types.models import ModelId, ModelMetadata
from shared.types.tasks import (
    ChatCompletionTask,
    TaskId,
    TaskStatus,
    TaskType,
)
from shared.types.worker.common import InstanceId
from shared.types.worker.instances import Instance, InstanceStatus
from shared.types.worker.runners import RunnerId, ShardAssignments
from shared.types.worker.shards import PipelineShardMetadata
from worker.tests.constants import (
    COMMAND_1_ID,
    INSTANCE_1_ID,
    MODEL_A_ID,
    NODE_A,
    RUNNER_1_ID,
    TASK_1_ID,
)


@pytest.fixture
def user_message():
    """Override this fixture in tests to customize the message"""
    return "Hello, how are you?"

@pytest.fixture
def logger() -> Logger:
    return getLogger("test_logger")

@pytest.fixture
async def model_meta() -> ModelMetadata:
    return await get_model_meta('mlx-community/Llama-3.2-1B-Instruct-4bit')

@pytest.fixture
def hosts():
    def _hosts(count: int, offset: int = 0) -> list[Host]:
        return [
            Host(
                ip=IPv4Address("127.0.0.1"),
                port=5000 + offset + i,
            )
            for i in range(count)
        ]

    return _hosts

@pytest.fixture
def pipeline_shard_meta(model_meta: ModelMetadata) -> Callable[[int, int], PipelineShardMetadata]:
    def _pipeline_shard_meta(
        num_nodes: int = 1, device_rank: int = 0
    ) -> PipelineShardMetadata:
        total_layers = model_meta.n_layers
        layers_per_node = total_layers // num_nodes
        start_layer = device_rank * layers_per_node
        end_layer = (
            start_layer + layers_per_node
            if device_rank < num_nodes - 1
            else total_layers
        )

        return PipelineShardMetadata(
            model_meta=model_meta,
            device_rank=device_rank,
            n_layers=total_layers,
            start_layer=start_layer,
            end_layer=end_layer,
            world_size=num_nodes,
        )

    return _pipeline_shard_meta

@pytest.fixture
def instance(pipeline_shard_meta: Callable[[int, int], PipelineShardMetadata], hosts: Callable[[int], list[Host]]):
    from typing import Optional

    def _instance(
        instance_id: Optional[InstanceId] = None,
        node_id: Optional[NodeId] = None,
        runner_id: Optional[RunnerId] = None,
        model_id: Optional[ModelId] = None,
    ) -> Instance:
        resolved_instance_id = instance_id if instance_id is not None else INSTANCE_1_ID
        resolved_node_id = node_id if node_id is not None else NODE_A
        resolved_runner_id = runner_id if runner_id is not None else RUNNER_1_ID
        resolved_model_id = model_id if model_id is not None else MODEL_A_ID

        shard_assignments = ShardAssignments(
            model_id=resolved_model_id,
            runner_to_shard={
                resolved_runner_id: pipeline_shard_meta(1, 0)
            },
            node_to_runner={resolved_node_id: resolved_runner_id}
        )

        return Instance(
            instance_id=resolved_instance_id,
            instance_type=InstanceStatus.ACTIVE,
            shard_assignments=shard_assignments,
            hosts=hosts(1)
        )
    return _instance

@pytest.fixture
def completion_create_params(user_message: str) -> ChatCompletionTaskParams:
    """Creates ChatCompletionParams with the given message"""
    return ChatCompletionTaskParams(
        model="gpt-4",
        messages=[ChatCompletionMessage(role="user", content=user_message)],
        stream=True,
    )

@pytest.fixture
def chat_completion_task(completion_create_params: ChatCompletionTaskParams):
    def _chat_completion_task(instance_id: Optional[InstanceId] = None, task_id: Optional[TaskId] = None) -> ChatCompletionTask:
        if instance_id is None:
            instance_id = INSTANCE_1_ID
        if task_id is None:
            task_id = TASK_1_ID
        return ChatCompletionTask(
            task_id=task_id,
            command_id=COMMAND_1_ID,
            instance_id=instance_id,
            task_type=TaskType.CHAT_COMPLETION,
            task_status=TaskStatus.PENDING,
            task_params=completion_create_params
        )
    return _chat_completion_task


