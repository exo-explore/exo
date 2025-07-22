import asyncio
import uuid
from logging import Logger, getLogger
from pathlib import Path
from typing import Callable

import pytest

from shared.types.common import NodeId
from shared.types.models import ModelId, ModelMetadata
from shared.types.state import State
from shared.types.tasks.common import (
    ChatCompletionMessage,
    ChatCompletionTaskParams,
    Task,
    TaskId,
    TaskStatus,
    TaskType,
)
from shared.types.worker.common import InstanceId, NodeStatus
from shared.types.worker.instances import Instance, InstanceParams, TypeOfInstance
from shared.types.worker.mlx import Host
from shared.types.worker.ops import (
    AssignRunnerOp,
    RunnerUpOp,
)
from shared.types.worker.runners import RunnerId, ShardAssignments
from shared.types.worker.shards import PipelineShardMetadata
from worker.main import Worker


@pytest.fixture
def model_meta() -> ModelMetadata:
    # return _get_model_meta('mlx-community/Llama-3.2-1B-Instruct-4bit') # we can't do this! as it's an async function :(
    return ModelMetadata(
        model_id='mlx-community/Llama-3.2-1B-Instruct-4bit',
        pretty_name='llama3.2',
        storage_size_kilobytes=10**6,
        n_layers=16
    )


@pytest.fixture
def pipeline_shard_meta(model_meta: ModelMetadata, tmp_path: Path) -> Callable[[int, int], PipelineShardMetadata]:
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
            model_meta=model_meta,
            device_rank=device_rank,
            n_layers=total_layers,
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
def completion_create_params(user_message: str) -> ChatCompletionTaskParams:
    """Creates ChatCompletionParams with the given message"""
    return ChatCompletionTaskParams(
        model="gpt-4",
        messages=[ChatCompletionMessage(role="user", content=user_message)],
        stream=True,
    )

@pytest.fixture
def chat_completion_task(completion_create_params: ChatCompletionTaskParams) -> Task:
    """Creates a ChatCompletionTask directly for serdes testing"""
    return Task(task_id=TaskId(), instance_id=InstanceId(), task_type=TaskType.ChatCompletion, task_status=TaskStatus.Pending, task_params=completion_create_params)

@pytest.fixture
def chat_task(
    completion_create_params: ChatCompletionTaskParams,
) -> Task:
    """Creates the final Task object"""
    return Task(
        task_id=TaskId(),
        instance_id=InstanceId(),
        task_type=TaskType.ChatCompletion,
        task_status=TaskStatus.Pending,
        task_params=completion_create_params,
    )

@pytest.fixture
def state():
    node_status={
        NodeId(uuid.uuid4()): NodeStatus.Idle
    }

    return State(
        node_status=node_status,
    )

@pytest.fixture
def logger() -> Logger:
    return getLogger("test_logger")

@pytest.fixture
def instance(pipeline_shard_meta: Callable[[int, int], PipelineShardMetadata], hosts_one: list[Host]):
    def _instance(node_id: NodeId) -> Instance:
        model_id = ModelId(uuid.uuid4())
        runner_id = RunnerId(uuid.uuid4())        

        shard_assignments = ShardAssignments(
            model_id=model_id,
            runner_to_shard={
                runner_id: pipeline_shard_meta(1, 0)
            },
            node_to_runner={node_id: runner_id}
        )
        
        instance_params = InstanceParams(
            shard_assignments=shard_assignments,
            hosts=hosts_one
        )
        
        return Instance(
            instance_id=InstanceId(uuid.uuid4()),
            instance_params=instance_params,
            instance_type=TypeOfInstance.ACTIVE
        )
    return _instance

@pytest.fixture
def worker(state: State, logger: Logger):
    return Worker(NodeId(uuid.uuid4()), state, logger)

@pytest.fixture
async def worker_with_assigned_runner(worker: Worker, instance: Callable[[NodeId], Instance]):
    """Fixture that provides a worker with an already assigned runner."""
    await worker.start()
    await asyncio.sleep(0.01)
    
    instance_obj: Instance = instance(worker.node_id)
    
    # Extract runner_id from shard assignments
    runner_id = next(iter(instance_obj.instance_params.shard_assignments.runner_to_shard))
    
    # Assign the runner
    assign_op = AssignRunnerOp(
        runner_id=runner_id,
        shard_metadata=instance_obj.instance_params.shard_assignments.runner_to_shard[runner_id],
        hosts=instance_obj.instance_params.hosts,
        instance_id=instance_obj.instance_id,
    )
    
    async for _ in worker._execute_op(assign_op):  # type: ignore[misc]
        pass
    
    return worker, runner_id, instance_obj

@pytest.fixture
async def worker_with_running_runner(worker_with_assigned_runner: tuple[Worker, RunnerId, Instance]):
    """Fixture that provides a worker with an already assigned runner."""
    worker, runner_id, instance_obj = worker_with_assigned_runner

    runner_up_op = RunnerUpOp(runner_id=runner_id)
    async for _ in worker._execute_op(runner_up_op):  # type: ignore[misc]
        pass

    # Is the runner actually running?
    supervisor = next(iter(worker.assigned_runners.values())).runner
    assert supervisor is not None
    assert supervisor.healthy

    return worker, runner_id, instance_obj