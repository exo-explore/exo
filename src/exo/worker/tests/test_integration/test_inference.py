import asyncio
from typing import Callable

import pytest
from anyio import create_task_group

from exo.shared.types.api import ChatCompletionMessage, ChatCompletionTaskParams
from exo.shared.types.common import CommandId, Host, NodeId
from exo.shared.types.events import (
    InstanceCreated,
    InstanceDeleted,
    TaskCreated,
)
from exo.shared.types.models import ModelId
from exo.shared.types.tasks import (
    ChatCompletionTask,
    Task,
    TaskId,
    TaskStatus,
    TaskType,
)
from exo.shared.types.worker.common import InstanceId, RunnerId
from exo.shared.types.worker.instances import (
    Instance,
    InstanceStatus,
    ShardAssignments,
)
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.main import Worker
from exo.worker.tests.constants import (
    INSTANCE_1_ID,
    MASTER_NODE_ID,
    NODE_A,
    NODE_B,
    RUNNER_1_ID,
    RUNNER_2_ID,
    TASK_1_ID,
)
from exo.worker.tests.worker_management import (
    WorkerMailbox,
    read_streaming_response,
)


@pytest.fixture
def user_message():
    """Override this fixture in tests to customize the message"""
    return "What's the capital of Japan?"


async def test_runner_inference(
    instance: Callable[[InstanceId, NodeId, RunnerId], Instance],
    chat_completion_task: Callable[[InstanceId, TaskId], Task],
    worker_and_mailbox: tuple[Worker, WorkerMailbox],
):
    worker, global_events = worker_and_mailbox
    async with create_task_group() as tg:
        tg.start_soon(worker.run)
        instance_value: Instance = instance(INSTANCE_1_ID, NODE_A, RUNNER_1_ID)
        instance_value.instance_type = InstanceStatus.ACTIVE

        task: Task = chat_completion_task(INSTANCE_1_ID, TASK_1_ID)
        await global_events.append_events(
            [
                InstanceCreated(
                    instance=instance_value,
                ),
                TaskCreated(task_id=task.task_id, task=task),
            ],
            origin=MASTER_NODE_ID,
        )

        # TODO: This needs to get fixed - sometimes it misses the 'starting' event.
        (
            seen_task_started,
            seen_task_finished,
            response_string,
            _,
        ) = await read_streaming_response(global_events)

        assert seen_task_started
        assert seen_task_finished
        assert "tokyo" in response_string.lower()

        await global_events.append_events(
            [
                InstanceDeleted(
                    instance_id=instance_value.instance_id,
                ),
            ],
            origin=MASTER_NODE_ID,
        )

        await asyncio.sleep(0.3)
        worker.shutdown()
        # TODO: Ensure this is sufficient, or add mechanism to fail the test gracefully if workers do not shutdown properly.


async def test_2_runner_inference(
    pipeline_shard_meta: Callable[[int, int], PipelineShardMetadata],
    hosts: Callable[[int], list[Host]],
    chat_completion_task: Callable[[InstanceId, TaskId], Task],
    two_workers_with_shared_mailbox: tuple[Worker, Worker, WorkerMailbox],
):
    worker1, worker2, global_events = two_workers_with_shared_mailbox
    async with create_task_group() as tg:
        tg.start_soon(worker1.run)
        tg.start_soon(worker2.run)
        ## Instance
        model_id = ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit")

        shard_assignments = ShardAssignments(
            model_id=model_id,
            runner_to_shard={
                RUNNER_1_ID: pipeline_shard_meta(2, 0),
                RUNNER_2_ID: pipeline_shard_meta(2, 1),
            },
            node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
        )

        instance = Instance(
            instance_id=INSTANCE_1_ID,
            instance_type=InstanceStatus.ACTIVE,
            shard_assignments=shard_assignments,
            hosts=hosts(2),
        )

        task = chat_completion_task(INSTANCE_1_ID, TASK_1_ID)
        await global_events.append_events(
            [
                InstanceCreated(instance=instance),
                TaskCreated(task_id=task.task_id, task=task),
            ],
            origin=MASTER_NODE_ID,
        )

        (
            seen_task_started,
            seen_task_finished,
            response_string,
            _,
        ) = await read_streaming_response(global_events)

        assert seen_task_started
        assert seen_task_finished
        assert "tokyo" in response_string.lower()

        _ = global_events.collect()
        await asyncio.sleep(1.0)
        events = global_events.collect()
        assert len(events) == 0

        await global_events.append_events(
            [
                InstanceDeleted(
                    instance_id=instance.instance_id,
                ),
            ],
            origin=MASTER_NODE_ID,
        )

        await asyncio.sleep(2.0)
        worker1.shutdown()
        worker2.shutdown()
        # TODO: Ensure this is sufficient, or add mechanism to fail the test gracefully if workers do not shutdown properly.


# TODO: Multi message parallel
async def test_2_runner_multi_message(
    pipeline_shard_meta: Callable[[int, int], PipelineShardMetadata],
    hosts: Callable[[int], list[Host]],
    two_workers_with_shared_mailbox: tuple[Worker, Worker, WorkerMailbox],
):
    worker1, worker2, global_events = two_workers_with_shared_mailbox
    async with create_task_group() as tg:
        tg.start_soon(worker1.run)
        tg.start_soon(worker2.run)

        ## Instance
        model_id = ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit")

        shard_assignments = ShardAssignments(
            model_id=model_id,
            runner_to_shard={
                RUNNER_1_ID: pipeline_shard_meta(2, 0),
                RUNNER_2_ID: pipeline_shard_meta(2, 1),
            },
            node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
        )

        instance = Instance(
            instance_id=INSTANCE_1_ID,
            instance_type=InstanceStatus.ACTIVE,
            shard_assignments=shard_assignments,
            hosts=hosts(2),
        )

        # Task - we have three messages here, which is what the task is about

        completion_create_params = ChatCompletionTaskParams(
            model="gpt-4",
            messages=[
                ChatCompletionMessage(
                    role="user", content="What is the capital of France?"
                ),
                ChatCompletionMessage(
                    role="assistant", content="The capital of France is Paris."
                ),
                ChatCompletionMessage(
                    role="user",
                    content="Ok great. Now write me a haiku about what you can do there.",
                ),
            ],
            stream=True,
        )

        task = ChatCompletionTask(
            task_id=TASK_1_ID,
            command_id=CommandId(),
            instance_id=INSTANCE_1_ID,
            task_type=TaskType.CHAT_COMPLETION,
            task_status=TaskStatus.PENDING,
            task_params=completion_create_params,
        )

        await global_events.append_events(
            [
                InstanceCreated(instance=instance),
                TaskCreated(task_id=task.task_id, task=task),
            ],
            origin=MASTER_NODE_ID,
        )

        (
            seen_task_started,
            seen_task_finished,
            response_string,
            _,
        ) = await read_streaming_response(global_events)

        assert seen_task_started
        assert seen_task_finished
        assert any(
            keyword in response_string.lower()
            for keyword in ("kiss", "paris", "art", "love")
        )

        _ = global_events.collect()
        await asyncio.sleep(1.0)
        events = global_events.collect()
        assert len(events) == 0

        await global_events.append_events(
            [
                InstanceDeleted(
                    instance_id=instance.instance_id,
                ),
            ],
            origin=MASTER_NODE_ID,
        )

        worker1.shutdown()
        worker2.shutdown()
        # TODO: Ensure this is sufficient, or add mechanism to fail the test gracefully if workers do not shutdown properly.
