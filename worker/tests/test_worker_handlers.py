## Tests for worker state handlers

import asyncio
from typing import Callable

import pytest

from shared.types.common import NodeId
from shared.types.events.chunks import TokenChunk, TokenChunkData
from shared.types.events.events import ChunkGenerated, RunnerStatusUpdated
from shared.types.events.registry import Event
from shared.types.tasks.common import Task
from shared.types.worker.common import RunnerId
from shared.types.worker.instances import Instance
from shared.types.worker.ops import (
    AssignRunnerOp,
    DownloadOp,
    ExecuteTaskOp,
    RunnerDownOp,
    RunnerUpOp,
    UnassignRunnerOp,
)
from shared.types.worker.runners import (
    FailedRunnerStatus,
    LoadedRunnerStatus,
    ReadyRunnerStatus,
    RunningRunnerStatus,
)
from worker.main import Worker


@pytest.fixture
def user_message():
    """Override the default message to ask about France's capital"""
    return "What, according to Douglas Adams, is the meaning of life, the universe and everything?"

@pytest.mark.asyncio
async def test_assign_op(worker: Worker, instance: Callable[[NodeId], Instance]):
    await worker.start()
    await asyncio.sleep(0.01)

    instance_obj: Instance = instance(worker.node_id)
    runner_id: RunnerId | None = None
    for x in instance_obj.instance_params.shard_assignments.runner_to_shard:
        runner_id = x
    assert runner_id is not None

    assign_op = AssignRunnerOp(
        runner_id=runner_id,
        shard_metadata=instance_obj.instance_params.shard_assignments.runner_to_shard[runner_id],
        hosts=instance_obj.instance_params.hosts,
        instance_id=instance_obj.instance_id,
    )

    events: list[Event] = []

    async for event in worker._execute_op(assign_op):  # type: ignore[misc]
        events.append(event)

    # We should have a status update saying 'starting'.
    assert len(events) == 1
    assert isinstance(events[0], RunnerStatusUpdated)
    assert isinstance(events[0].runner_status, ReadyRunnerStatus)

    # And the runner should be assigned
    assert runner_id in worker.assigned_runners
    assert isinstance(worker.assigned_runners[runner_id].status, ReadyRunnerStatus)

@pytest.mark.asyncio
async def test_unassign_op(worker_with_assigned_runner: tuple[Worker, RunnerId, Instance]):
    worker, runner_id, _ = worker_with_assigned_runner

    unassign_op = UnassignRunnerOp(
        runner_id=runner_id
    )

    events: list[Event] = []

    async for event in worker._execute_op(unassign_op):  # type: ignore[misc]
        events.append(event)

    # We should have no assigned runners and no events were emitted
    assert len(worker.assigned_runners) == 0
    assert len(events) == 0

@pytest.mark.asyncio
async def test_runner_up_op(worker_with_assigned_runner: tuple[Worker, RunnerId, Instance], chat_task: Task):
    worker, runner_id, _ = worker_with_assigned_runner

    runner_up_op = RunnerUpOp(runner_id=runner_id)

    events: list[Event] = []
    async for event in worker._execute_op(runner_up_op):  # type: ignore[misc]
        events.append(event)

    assert len(events) == 1
    assert isinstance(events[0], RunnerStatusUpdated)
    assert isinstance(events[0].runner_status, LoadedRunnerStatus)

    # Is the runner actually running?
    supervisor = next(iter(worker.assigned_runners.values())).runner
    assert supervisor is not None
    assert supervisor.healthy

    full_response = ''

    async for chunk in supervisor.stream_response(task=chat_task):
        if isinstance(chunk, TokenChunk):
            full_response += chunk.chunk_data.text

    assert "42" in full_response.lower(), (
        f"Expected '42' in response, but got: {full_response}"
    )

    runner = worker.assigned_runners[runner_id].runner
    assert runner is not None
    await runner.astop() # Neat cleanup.

@pytest.mark.asyncio
async def test_runner_down_op(worker_with_running_runner: tuple[Worker, RunnerId, Instance]):
    worker, runner_id, _ = worker_with_running_runner

    runner_down_op = RunnerDownOp(runner_id=runner_id)
    events: list[Event] = []
    async for event in worker._execute_op(runner_down_op):  # type: ignore[misc]
        events.append(event)

    assert len(events) == 1
    assert isinstance(events[0], RunnerStatusUpdated)
    assert isinstance(events[0].runner_status, ReadyRunnerStatus)

@pytest.mark.asyncio
async def test_download_op(worker_with_assigned_runner: tuple[Worker, RunnerId, Instance]):
    worker, runner_id, instance_obj = worker_with_assigned_runner

    print(f'{worker.assigned_runners=}')

    download_op = DownloadOp(
        instance_id=instance_obj.instance_id,
        runner_id=runner_id,
        shard_metadata=instance_obj.instance_params.shard_assignments.runner_to_shard[runner_id],
        hosts=instance_obj.instance_params.hosts,
    )

    events: list[Event] = []

    async for event in worker._execute_op(download_op):  # type: ignore[misc]
        events.append(event)

    # Should give download status and then a final download status with DownloadCompleted
    print(events)

@pytest.mark.asyncio
async def test_execute_task_op(
    worker_with_running_runner: tuple[Worker, RunnerId, Instance], 
    chat_task: Task):
    worker, runner_id, _ = worker_with_running_runner

    execute_task_op = ExecuteTaskOp(
        runner_id=runner_id,
        task=chat_task
    )

    events: list[Event] = []
    async for event in worker._execute_op(execute_task_op): # type: ignore[misc]
        events.append(event)

    assert len(events) > 20

    assert isinstance(events[0], RunnerStatusUpdated)
    assert isinstance(events[0].runner_status, RunningRunnerStatus)

    assert isinstance(events[-1], RunnerStatusUpdated)
    assert isinstance(events[-1].runner_status, LoadedRunnerStatus) # It should not have failed.

    gen_events: list[ChunkGenerated] = [x for x in events if isinstance(x, ChunkGenerated)]
    text_chunks: list[TokenChunkData] = [x.chunk.chunk_data for x in gen_events if isinstance(x.chunk.chunk_data, TokenChunkData)]
    assert len(text_chunks) == len(events) - 2
    
    output_text = ''.join([x.text for x in text_chunks])
    assert '42' in output_text

    runner = worker.assigned_runners[runner_id].runner
    assert runner is not None
    await runner.astop() # Neat cleanup.

@pytest.mark.asyncio
async def test_execute_task_fails(
    worker_with_running_runner: tuple[Worker, RunnerId, Instance], 
    chat_task: Task):
    worker, runner_id, _ = worker_with_running_runner

    messages = chat_task.task_params.messages
    messages[0].content = 'Artificial prompt: EXO RUNNER MUST FAIL'

    execute_task_op = ExecuteTaskOp(
        runner_id=runner_id,
        task=chat_task
    )

    events: list[Event] = []
    async for event in worker._execute_op(execute_task_op): # type: ignore[misc]
        events.append(event)

    assert len(events) == 2

    assert isinstance(events[0], RunnerStatusUpdated)
    assert isinstance(events[0].runner_status, RunningRunnerStatus) # It tried to start.

    assert isinstance(events[-1], RunnerStatusUpdated)
    assert isinstance(events[-1].runner_status, FailedRunnerStatus) # It should have failed. 