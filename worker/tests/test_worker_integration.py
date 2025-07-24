import asyncio
from typing import Awaitable, Callable, Final

import pytest

from shared.db.sqlite.connector import AsyncSQLiteEventStorage
from shared.types.common import NodeId
from shared.types.events import (
    InstanceCreated,
    InstanceDeleted,
    RunnerDeleted,
    RunnerStatusUpdated,
)
from shared.types.events.chunks import TokenChunk
from shared.types.models import ModelId
from shared.types.tasks import Task, TaskId
from shared.types.worker.common import InstanceId, RunnerId
from shared.types.worker.instances import Instance, TypeOfInstance
from shared.types.worker.runners import (
    LoadedRunnerStatus,
    ReadyRunnerStatus,
    # RunningRunnerStatus,
)
from worker.main import Worker

MASTER_NODE_ID = NodeId("ffffffff-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
NODE_A: Final[NodeId] = NodeId("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
NODE_B: Final[NodeId] = NodeId("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")

# Define constant IDs for deterministic test cases
RUNNER_1_ID: Final[RunnerId] = RunnerId()
INSTANCE_1_ID: Final[InstanceId] = InstanceId()
RUNNER_2_ID: Final[RunnerId] = RunnerId()
INSTANCE_2_ID: Final[InstanceId] = InstanceId()
MODEL_A_ID: Final[ModelId] = 'mlx-community/Llama-3.2-1B-Instruct-4bit'
MODEL_B_ID: Final[ModelId] = 'mlx-community/Llama-3.2-1B-Instruct-4bit'
TASK_1_ID: Final[TaskId] = TaskId()

@pytest.fixture
def user_message():
    return "What is the capital of Japan?"

async def test_runner_assigned(
    worker_running: Callable[[NodeId], Awaitable[tuple[Worker, AsyncSQLiteEventStorage]]], 
    instance: Callable[[NodeId, RunnerId], Instance]
    ):

    worker, global_events = await worker_running(NODE_A)

    print(worker)

    instance_value: Instance = instance(NODE_A, RUNNER_1_ID)
    instance_value.instance_type = TypeOfInstance.INACTIVE

    await global_events.append_events(
        [
            InstanceCreated(
                instance_id=instance_value.instance_id,
                instance_params=instance_value.instance_params,
                instance_type=instance_value.instance_type
            )
        ], 
        origin=MASTER_NODE_ID
    )

    await asyncio.sleep(0.1)

    # Ensure the worker has taken the correct action
    assert len(worker.assigned_runners) == 1
    assert RUNNER_1_ID in worker.assigned_runners
    assert isinstance(worker.assigned_runners[RUNNER_1_ID].status, ReadyRunnerStatus)

    # Ensure the correct events have been emitted
    events = await global_events.get_events_since(0)
    assert len(events) == 2
    assert isinstance(events[1].event, RunnerStatusUpdated)
    assert isinstance(events[1].event.runner_status, ReadyRunnerStatus)

    # Ensure state is correct
    assert isinstance(worker.state.runners[RUNNER_1_ID], ReadyRunnerStatus)

async def test_runner_assigned_active(
    worker_running: Callable[[NodeId], Awaitable[tuple[Worker, AsyncSQLiteEventStorage]]],
    instance: Callable[[NodeId, RunnerId], Instance],
    chat_completion_task: Task
    ):
    worker, global_events = await worker_running(NODE_A)

    instance_value: Instance = instance(NODE_A, RUNNER_1_ID)
    instance_value.instance_type = TypeOfInstance.ACTIVE

    await global_events.append_events(
        [
            InstanceCreated(
                instance_id=instance_value.instance_id,
                instance_params=instance_value.instance_params,
                instance_type=instance_value.instance_type
            )
        ], 
        origin=MASTER_NODE_ID
    )

    await asyncio.sleep(0.1)

    assert len(worker.assigned_runners) == 1
    assert RUNNER_1_ID in worker.assigned_runners
    assert isinstance(worker.assigned_runners[RUNNER_1_ID].status, LoadedRunnerStatus)

    # Ensure the correct events have been emitted
    events = await global_events.get_events_since(0)
    assert len(events) == 3
    assert isinstance(events[2].event, RunnerStatusUpdated)
    assert isinstance(events[2].event.runner_status, LoadedRunnerStatus)

    # Ensure state is correct
    assert isinstance(worker.state.runners[RUNNER_1_ID], LoadedRunnerStatus)
    
    # Ensure that the runner has been created and it can stream tokens.
    supervisor = next(iter(worker.assigned_runners.values())).runner
    assert supervisor is not None
    assert supervisor.healthy

    full_response = ''

    async for chunk in supervisor.stream_response(task=chat_completion_task):
        if isinstance(chunk, TokenChunk):
            full_response += chunk.text

    assert "tokyo" in full_response.lower(), (
        f"Expected 'Tokyo' in response, but got: {full_response}"
    )

async def test_runner_assigned_wrong_node(
    worker_running: Callable[[NodeId], Awaitable[tuple[Worker, AsyncSQLiteEventStorage]]],
    instance: Callable[[NodeId, RunnerId], Instance]
    ):
    worker, global_events = await worker_running(NODE_A)

    instance_value = instance(NODE_B, RUNNER_1_ID)

    await global_events.append_events(
        [
            InstanceCreated(
                instance_id=instance_value.instance_id,
                instance_params=instance_value.instance_params,
                instance_type=instance_value.instance_type
            )
        ], 
        origin=MASTER_NODE_ID
    )

    await asyncio.sleep(0.1)

    assert len(worker.assigned_runners) == 0

    # Ensure the correct events have been emitted
    events = await global_events.get_events_since(0)
    assert len(events) == 1
    # No RunnerStatusUpdated event should be emitted

    # Ensure state is correct
    assert len(worker.state.runners) == 0

async def test_runner_unassigns(
    worker_running: Callable[[NodeId], Awaitable[tuple[Worker, AsyncSQLiteEventStorage]]],
    instance: Callable[[NodeId, RunnerId], Instance]
    ):
    worker, global_events = await worker_running(NODE_A)

    instance_value: Instance = instance(NODE_A, RUNNER_1_ID)
    instance_value.instance_type = TypeOfInstance.ACTIVE

    await global_events.append_events(
        [
            InstanceCreated(
                instance_id=instance_value.instance_id,
                instance_params=instance_value.instance_params,
                instance_type=instance_value.instance_type
            )
        ], 
        origin=MASTER_NODE_ID
    )

    await asyncio.sleep(0.1)

    # already tested by test_runner_assigned_active
    assert len(worker.assigned_runners) == 1
    assert RUNNER_1_ID in worker.assigned_runners
    assert isinstance(worker.assigned_runners[RUNNER_1_ID].status, LoadedRunnerStatus)

    # Ensure the correct events have been emitted (creation)
    events = await global_events.get_events_since(0)
    assert len(events) == 3
    assert isinstance(events[2].event, RunnerStatusUpdated)
    assert isinstance(events[2].event.runner_status, LoadedRunnerStatus)

    # Ensure state is correct
    print(worker.state)
    assert isinstance(worker.state.runners[RUNNER_1_ID], LoadedRunnerStatus)

    await global_events.append_events(
        [
            InstanceDeleted(instance_id=instance_value.instance_id)
        ], 
        origin=MASTER_NODE_ID
    )

    await asyncio.sleep(0.3)

    print(worker.state)
    assert len(worker.assigned_runners) == 0

    # Ensure the correct events have been emitted (deletion)
    events = await global_events.get_events_since(0)
    assert isinstance(events[-1].event, RunnerDeleted)
    # After deletion, runner should be removed from state.runners
    assert len(worker.state.runners) == 0