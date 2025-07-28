import asyncio
from logging import Logger
from typing import Awaitable, Callable, Final

import pytest

# TaskStateUpdated and ChunkGenerated are used in test_worker_integration_utils.py
from shared.db.sqlite.connector import AsyncSQLiteEventStorage
from shared.db.sqlite.event_log_manager import EventLogConfig, EventLogManager
from shared.types.common import Host, NodeId
from shared.types.events import (
    InstanceCreated,
    InstanceDeleted,
    RunnerDeleted,
    RunnerStatusUpdated,
    TaskCreated,
)
from shared.types.events.chunks import TokenChunk
from shared.types.models import ModelId
from shared.types.tasks import Task, TaskId
from shared.types.worker.common import InstanceId, RunnerId
from shared.types.worker.instances import (
    Instance,
    InstanceStatus,
    ShardAssignments,
)
from shared.types.worker.runners import (
    AssignedRunnerStatus,
    DownloadingRunnerStatus,
    # RunningRunnerStatus,
    FailedRunnerStatus,
    LoadedRunnerStatus,
    ReadyRunnerStatus,
)
from shared.types.worker.shards import PipelineShardMetadata
from worker.download.shard_downloader import NoopShardDownloader
from worker.main import AssignedRunner, Worker
from worker.tests.test_worker_integration_utils import read_streaming_response

MASTER_NODE_ID = NodeId("ffffffff-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
NODE_A: Final[NodeId] = NodeId("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
NODE_B: Final[NodeId] = NodeId("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")

# Define constant IDs for deterministic test cases
RUNNER_1_ID: Final[RunnerId] = RunnerId("11111111-1111-4111-8111-111111111111")
INSTANCE_1_ID: Final[InstanceId] = InstanceId("22222222-2222-4222-8222-222222222222")
RUNNER_2_ID: Final[RunnerId] = RunnerId("33333333-3333-4333-8333-333333333333")
INSTANCE_2_ID: Final[InstanceId] = InstanceId("44444444-4444-4444-8444-444444444444")
MODEL_A_ID: Final[ModelId] = 'mlx-community/Llama-3.2-1B-Instruct-4bit'
MODEL_B_ID: Final[ModelId] = 'mlx-community/Llama-3.2-1B-Instruct-4bit'
TASK_1_ID: Final[TaskId] = TaskId("55555555-5555-4555-8555-555555555555")
TASK_2_ID: Final[TaskId] = TaskId("66666666-6666-4666-8666-666666666666")

@pytest.fixture
def user_message():
    return "What is the capital of Japan?"

async def test_runner_assigned(
    worker_running: Callable[[NodeId], Awaitable[tuple[Worker, AsyncSQLiteEventStorage]]], 
    instance: Callable[[InstanceId, NodeId, RunnerId], Instance]
    ):

    worker, global_events = await worker_running(NODE_A)

    print(worker)

    instance_value: Instance = instance(INSTANCE_1_ID, NODE_A, RUNNER_1_ID)
    instance_value.instance_type = InstanceStatus.INACTIVE

    await global_events.append_events(
        [
            InstanceCreated(
                instance=instance_value
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
    print(events)
    assert len(events) >= 4 # len(events) is 4 if it's already downloaded. It is > 4 if there have to be download events.

    assert isinstance(events[1].event, RunnerStatusUpdated)
    assert isinstance(events[1].event.runner_status, AssignedRunnerStatus)
    assert isinstance(events[2].event, RunnerStatusUpdated)
    assert isinstance(events[2].event.runner_status, DownloadingRunnerStatus)
    assert isinstance(events[-1].event, RunnerStatusUpdated)
    assert isinstance(events[-1].event.runner_status, ReadyRunnerStatus)

    # Ensure state is correct
    assert isinstance(worker.state.runners[RUNNER_1_ID], ReadyRunnerStatus)

async def test_runner_assigned_active(
    worker_running: Callable[[NodeId], Awaitable[tuple[Worker, AsyncSQLiteEventStorage]]],
    instance: Callable[[InstanceId, NodeId, RunnerId], Instance],
    chat_completion_task: Callable[[InstanceId, TaskId], Task]
    ):
    worker, global_events = await worker_running(NODE_A)

    instance_value: Instance = instance(INSTANCE_1_ID, NODE_A, RUNNER_1_ID)
    instance_value.instance_type = InstanceStatus.ACTIVE

    await global_events.append_events(
        [
            InstanceCreated(
                instance=instance_value
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
    assert len(events) >= 5 # len(events) is 5 if it's already downloaded. It is > 5 if there have to be download events.
    assert isinstance(events[1].event, RunnerStatusUpdated)
    assert isinstance(events[1].event.runner_status, AssignedRunnerStatus)
    assert isinstance(events[2].event, RunnerStatusUpdated)
    assert isinstance(events[2].event.runner_status, DownloadingRunnerStatus)
    assert isinstance(events[-2].event, RunnerStatusUpdated)
    assert isinstance(events[-2].event.runner_status, ReadyRunnerStatus)
    assert isinstance(events[-1].event, RunnerStatusUpdated)
    assert isinstance(events[-1].event.runner_status, LoadedRunnerStatus)

    # Ensure state is correct
    assert isinstance(worker.state.runners[RUNNER_1_ID], LoadedRunnerStatus)
    
    # Ensure that the runner has been created and it can stream tokens.
    supervisor = next(iter(worker.assigned_runners.values())).runner
    assert supervisor is not None
    assert supervisor.healthy

    full_response = ''

    async for chunk in supervisor.stream_response(task=chat_completion_task(INSTANCE_1_ID, TASK_1_ID)):
        if isinstance(chunk, TokenChunk):
            full_response += chunk.text

    assert "tokyo" in full_response.lower(), (
        f"Expected 'Tokyo' in response, but got: {full_response}"
    )

async def test_runner_assigned_wrong_node(
    worker_running: Callable[[NodeId], Awaitable[tuple[Worker, AsyncSQLiteEventStorage]]],
    instance: Callable[[InstanceId, NodeId, RunnerId], Instance]
    ):
    worker, global_events = await worker_running(NODE_A)

    instance_value = instance(INSTANCE_1_ID, NODE_B, RUNNER_1_ID)

    await global_events.append_events(
        [
            InstanceCreated(
                instance=instance_value
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
    instance: Callable[[InstanceId, NodeId, RunnerId], Instance]
    ):
    worker, global_events = await worker_running(NODE_A)

    instance_value: Instance = instance(INSTANCE_1_ID, NODE_A, RUNNER_1_ID)
    instance_value.instance_type = InstanceStatus.ACTIVE

    await global_events.append_events(
        [
            InstanceCreated(
                instance=instance_value
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
    assert len(events) >= 5
    assert isinstance(events[-1].event, RunnerStatusUpdated)
    assert isinstance(events[-1].event.runner_status, LoadedRunnerStatus)

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

async def test_runner_inference(
    worker_running: Callable[[NodeId], Awaitable[tuple[Worker, AsyncSQLiteEventStorage]]],
    instance: Callable[[InstanceId, NodeId, RunnerId], Instance],
    chat_completion_task: Callable[[InstanceId, TaskId], Task]
    ):
    _worker, global_events = await worker_running(NODE_A)

    instance_value: Instance = instance(INSTANCE_1_ID, NODE_A, RUNNER_1_ID)
    instance_value.instance_type = InstanceStatus.ACTIVE

    task: Task = chat_completion_task(INSTANCE_1_ID, TASK_1_ID)
    await global_events.append_events(
        [
            InstanceCreated(
                instance=instance_value,
            ),
            TaskCreated(
                task_id=task.task_id,
                task=task
            )
        ], 
        origin=MASTER_NODE_ID
    )
    
    seen_task_started, seen_task_finished, response_string = await read_streaming_response(global_events)

    assert seen_task_started
    assert seen_task_finished
    assert 'tokyo' in response_string.lower()

    await global_events.append_events(
        [
            InstanceDeleted(
                instance_id=instance_value.instance_id,
            ),
        ], 
        origin=MASTER_NODE_ID
    )

    await asyncio.sleep(0.3)

async def test_2_runner_inference(
    logger: Logger,
    pipeline_shard_meta: Callable[[int, int], PipelineShardMetadata],
    hosts: Callable[[int], list[Host]],
    chat_completion_task: Callable[[InstanceId, TaskId], Task]
    ):
    event_log_manager = EventLogManager(EventLogConfig(), logger)
    await event_log_manager.initialize()
    shard_downloader = NoopShardDownloader()

    global_events = event_log_manager.global_events
    await global_events.delete_all_events()

    worker1 = Worker(NODE_A, logger=logger, shard_downloader=shard_downloader, worker_events=global_events, global_events=global_events)
    asyncio.create_task(worker1.run())

    worker2 = Worker(NODE_B, logger=logger, shard_downloader=shard_downloader, worker_events=global_events, global_events=global_events)
    asyncio.create_task(worker2.run())

    ## Instance
    model_id = ModelId('mlx-community/Llama-3.2-1B-Instruct-4bit')

    shard_assignments = ShardAssignments(
        model_id=model_id,
        runner_to_shard={
            RUNNER_1_ID: pipeline_shard_meta(2, 0),
            RUNNER_2_ID: pipeline_shard_meta(2, 1)
        },
        node_to_runner={
            NODE_A: RUNNER_1_ID,
            NODE_B: RUNNER_2_ID
        }
    )
    
    instance = Instance(
        instance_id=INSTANCE_1_ID,
        instance_type=InstanceStatus.ACTIVE,
        shard_assignments=shard_assignments,
        hosts=hosts(2)
    )

    task = chat_completion_task(INSTANCE_1_ID, TASK_1_ID)
    await global_events.append_events(
        [
            InstanceCreated(
                instance=instance
            ),
            TaskCreated(
                task_id=task.task_id,
                task=task
            )
        ], 
        origin=MASTER_NODE_ID
    )

    seen_task_started, seen_task_finished, response_string = await read_streaming_response(global_events)    

    assert seen_task_started
    assert seen_task_finished
    assert 'tokyo' in response_string.lower()


    idx = await global_events.get_last_idx()
    await asyncio.sleep(1.0)
    events = await global_events.get_events_since(idx)
    assert len(events) == 0

    await global_events.append_events(
        [
            InstanceDeleted(
                instance_id=instance.instance_id,
            ),
        ], 
        origin=MASTER_NODE_ID
    )

    await asyncio.sleep(2.0)



async def test_runner_respawn(
    logger: Logger,
    pipeline_shard_meta: Callable[[int, int], PipelineShardMetadata],
    hosts: Callable[[int], list[Host]],
    chat_completion_task: Callable[[InstanceId, TaskId], Task]
    ):
    event_log_manager = EventLogManager(EventLogConfig(), logger)
    await event_log_manager.initialize()
    shard_downloader = NoopShardDownloader()

    global_events = event_log_manager.global_events
    await global_events.delete_all_events()

    worker1 = Worker(NODE_A, logger=logger, shard_downloader=shard_downloader, worker_events=global_events, global_events=global_events)
    asyncio.create_task(worker1.run())

    worker2 = Worker(NODE_B, logger=logger, shard_downloader=shard_downloader, worker_events=global_events, global_events=global_events)
    asyncio.create_task(worker2.run())

    ## Instance
    model_id = ModelId('mlx-community/Llama-3.2-1B-Instruct-4bit')

    shard_assignments = ShardAssignments(
        model_id=model_id,
        runner_to_shard={
            RUNNER_1_ID: pipeline_shard_meta(2, 0),
            RUNNER_2_ID: pipeline_shard_meta(2, 1)
        },
        node_to_runner={
            NODE_A: RUNNER_1_ID,
            NODE_B: RUNNER_2_ID
        }
    )

    instance = Instance(
        instance_id=INSTANCE_1_ID,
        instance_type=InstanceStatus.ACTIVE,
        shard_assignments=shard_assignments,
        hosts=hosts(2)
    )

    task = chat_completion_task(INSTANCE_1_ID, TASK_1_ID)
    await global_events.append_events(
        [
            InstanceCreated(
                instance=instance
            ),
            TaskCreated(
                task_id=task.task_id,
                task=task
            )
        ], 
        origin=MASTER_NODE_ID
    )

    seen_task_started, seen_task_finished, response_string = await read_streaming_response(global_events)    

    assert seen_task_started
    assert seen_task_finished
    assert 'tokyo' in response_string.lower()

    await asyncio.sleep(0.1)

    idx = await global_events.get_last_idx()

    assigned_runner: AssignedRunner = worker1.assigned_runners[RUNNER_1_ID]
    assert assigned_runner.runner is not None
    assigned_runner.runner.runner_process.kill()
    
    # Wait for the process to actually be detected as dead or cleaned up
    for _ in range(100):  # Wait up to 1 second
        await asyncio.sleep(0.01)
        # The worker may clean up the runner (set to None) when it detects it's dead
        if assigned_runner.runner and not assigned_runner.runner.healthy:
            break
    else:
        raise AssertionError("Runner should have been detected as unhealthy or cleaned up after kill()")
    
    await asyncio.sleep(5.0)

    events = await global_events.get_events_since(idx)
    print(f'{events=}')
    # assert len(events) == 2
    assert isinstance(events[0].event, RunnerStatusUpdated)
    assert isinstance(events[0].event.runner_status, FailedRunnerStatus)

    assert isinstance(events[1].event, RunnerStatusUpdated)
    assert isinstance(events[1].event.runner_status, ReadyRunnerStatus)
    assert events[1].event.runner_id == RUNNER_2_ID

    assert isinstance(events[2].event, RunnerStatusUpdated)
    assert isinstance(events[2].event.runner_status, ReadyRunnerStatus)
    assert events[2].event.runner_id == RUNNER_1_ID

    print(worker1.state)
    print(worker2.state)

    for event in [events[3].event, events[4].event]:
        assert isinstance(event, RunnerStatusUpdated)
        assert isinstance(event.runner_status, LoadedRunnerStatus)

    task = chat_completion_task(INSTANCE_1_ID, TASK_2_ID)
    await global_events.append_events(
        [
            TaskCreated(
                task_id=task.task_id,
                task=task
            )
        ], 
        origin=MASTER_NODE_ID
    )

    seen_task_started, seen_task_finished, response_string = await read_streaming_response(global_events)    

    assert seen_task_started
    assert seen_task_finished
    assert 'tokyo' in response_string.lower()

    await asyncio.sleep(0.1)

    await global_events.append_events(
        [
            InstanceDeleted(
                instance_id=instance.instance_id,
            ),
        ], 
        origin=MASTER_NODE_ID
    )

    await asyncio.sleep(1.0)