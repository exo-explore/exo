from typing import Callable

import pytest

from exo.shared.types.common import NodeId
from exo.shared.types.events import (
    ChunkGenerated,
    RunnerDeleted,
    RunnerStatusUpdated,
    TaskStateUpdated,
)
from exo.shared.types.events.chunks import TokenChunk
from exo.shared.types.tasks import ChatCompletionTask, TaskStatus
from exo.shared.types.worker.common import RunnerId
from exo.shared.types.worker.instances import Instance, InstanceId
from exo.shared.types.worker.ops import (
    AssignRunnerOp,
    ExecuteTaskOp,
    RunnerDownOp,
    RunnerUpOp,
    UnassignRunnerOp,
)
from exo.shared.types.worker.runners import (
    DownloadingRunnerStatus,
    InactiveRunnerStatus,
    LoadedRunnerStatus,
    RunningRunnerStatus,
)
from exo.worker.main import Worker
from exo.worker.tests.constants import (
    RUNNER_1_ID,
)
from exo.worker.tests.test_handlers.utils import read_events_op


@pytest.mark.asyncio
async def test_assign_op(
    worker: Worker, instance: Callable[[InstanceId, NodeId, RunnerId], Instance]
):
    instance_obj: Instance = instance(InstanceId(), worker.node_id, RUNNER_1_ID)

    assign_op = AssignRunnerOp(
        runner_id=RUNNER_1_ID,
        shard_metadata=instance_obj.shard_assignments.runner_to_shard[RUNNER_1_ID],
        hosts=instance_obj.hosts,
        instance_id=instance_obj.instance_id,
    )

    events = await read_events_op(worker, assign_op)

    # We should have a status update saying 'starting'.
    assert len(events) == 2
    assert isinstance(events[0], RunnerStatusUpdated)
    assert isinstance(events[0].runner_status, DownloadingRunnerStatus)
    assert isinstance(events[1], RunnerStatusUpdated)
    assert isinstance(events[1].runner_status, InactiveRunnerStatus)

    # And the runner should be assigned
    assert RUNNER_1_ID in worker.assigned_runners
    assert isinstance(worker.assigned_runners[RUNNER_1_ID].status, InactiveRunnerStatus)


@pytest.mark.asyncio
async def test_unassign_op(worker_with_assigned_runner: tuple[Worker, Instance]):
    worker, _ = worker_with_assigned_runner

    unassign_op = UnassignRunnerOp(runner_id=RUNNER_1_ID)

    events = await read_events_op(worker, unassign_op)

    # We should have no assigned runners and no events were emitted
    assert len(worker.assigned_runners) == 0
    assert len(events) == 1
    assert isinstance(events[0], RunnerDeleted)


@pytest.mark.asyncio
async def test_runner_up_op(
    worker_with_assigned_runner: tuple[Worker, Instance],
    chat_completion_task: Callable[[], ChatCompletionTask],
):
    worker, _ = worker_with_assigned_runner

    runner_up_op = RunnerUpOp(runner_id=RUNNER_1_ID)

    events = await read_events_op(worker, runner_up_op)

    assert len(events) == 1
    assert isinstance(events[0], RunnerStatusUpdated)
    assert isinstance(events[0].runner_status, LoadedRunnerStatus)

    # Is the runner actually running?
    supervisor = next(iter(worker.assigned_runners.values())).runner
    assert supervisor is not None
    assert supervisor.healthy

    full_response = ""

    async for chunk in supervisor.stream_response(task=chat_completion_task()):
        if isinstance(chunk, TokenChunk):
            full_response += chunk.text

    assert "42" in full_response.lower(), (
        f"Expected '42' in response, but got: {full_response}"
    )

    runner = worker.assigned_runners[RUNNER_1_ID].runner
    assert runner is not None
    await runner.astop()  # Neat cleanup.


@pytest.mark.asyncio
async def test_runner_down_op(worker_with_running_runner: tuple[Worker, Instance]):
    worker, _ = worker_with_running_runner

    runner_down_op = RunnerDownOp(runner_id=RUNNER_1_ID)
    events = await read_events_op(worker, runner_down_op)

    assert len(events) == 1
    assert isinstance(events[0], RunnerStatusUpdated)
    assert isinstance(events[0].runner_status, InactiveRunnerStatus)


@pytest.mark.asyncio
async def test_execute_task_op(
    worker_with_running_runner: tuple[Worker, Instance],
    chat_completion_task: Callable[[], ChatCompletionTask],
):
    worker, _ = worker_with_running_runner

    execute_task_op = ExecuteTaskOp(runner_id=RUNNER_1_ID, task=chat_completion_task())

    events = await read_events_op(worker, execute_task_op)

    assert len(events) > 20

    print(f"{events=}")

    assert isinstance(events[0], RunnerStatusUpdated)
    assert isinstance(events[0].runner_status, RunningRunnerStatus)

    assert isinstance(events[1], TaskStateUpdated)
    assert events[1].task_status == TaskStatus.RUNNING  # It tried to start.

    assert isinstance(events[-2], TaskStateUpdated)
    assert events[-2].task_status == TaskStatus.COMPLETE  # It tried to start.

    assert isinstance(events[-1], RunnerStatusUpdated)
    assert isinstance(
        events[-1].runner_status, LoadedRunnerStatus
    )  # It should not have failed.

    gen_events: list[ChunkGenerated] = [
        x for x in events if isinstance(x, ChunkGenerated)
    ]
    text_chunks: list[TokenChunk] = [
        x.chunk for x in gen_events if isinstance(x.chunk, TokenChunk)
    ]
    assert len(text_chunks) == len(events) - 4

    output_text = "".join([x.text for x in text_chunks])
    assert "42" in output_text

    runner = worker.assigned_runners[RUNNER_1_ID].runner
    assert runner is not None
    await runner.astop()  # Neat cleanup.
