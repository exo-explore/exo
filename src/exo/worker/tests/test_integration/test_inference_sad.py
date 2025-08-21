import asyncio
from collections.abc import AsyncGenerator
from types import CoroutineType
from typing import Any, Awaitable, Callable

import pytest
from _pytest.monkeypatch import MonkeyPatch

# TaskStateUpdated and ChunkGenerated are used in test_worker_integration_utils.py
from exo.shared.db.sqlite.connector import AsyncSQLiteEventStorage
from exo.shared.types.common import NodeId
from exo.shared.types.events import (
    ChunkGenerated,
    InstanceCreated,
    InstanceDeleted,
    RunnerStatusUpdated,
    TaskCreated,
    TaskStateUpdated,
)
from exo.shared.types.events._events import TaskFailed
from exo.shared.types.events.chunks import GenerationChunk, TokenChunk
from exo.shared.types.tasks import Task, TaskId, TaskStatus
from exo.shared.types.worker.common import InstanceId, RunnerId
from exo.shared.types.worker.instances import (
    Instance,
    InstanceStatus,
)
from exo.shared.types.worker.runners import FailedRunnerStatus
from exo.worker.main import Worker
from exo.worker.runner.runner_supervisor import RunnerSupervisor
from exo.worker.tests.constants import (
    INSTANCE_1_ID,
    MASTER_NODE_ID,
    NODE_A,
    RUNNER_1_ID,
    TASK_1_ID,
)
from exo.worker.tests.test_integration.integration_utils import until_event_with_timeout


@pytest.fixture
def user_message():
    """Override this fixture in tests to customize the message"""
    return "Who is the longest ruling monarch of England?"


async def test_stream_response_failed_always(
    monkeypatch: MonkeyPatch,
    worker_running: Callable[[NodeId], Awaitable[tuple[Worker, AsyncSQLiteEventStorage]]],
    instance: Callable[[InstanceId, NodeId, RunnerId], Instance],
    chat_completion_task: Callable[[InstanceId, TaskId], Task]
) -> None:
    _, global_events = await worker_running(NODE_A)

    instance_value: Instance = instance(INSTANCE_1_ID, NODE_A, RUNNER_1_ID)
    instance_value.instance_type = InstanceStatus.ACTIVE
    
    async def mock_stream_response(
        self: RunnerSupervisor,
        task: Task,
        request_started_callback: Callable[..., CoroutineType[Any, Any, None]] | None = None,
    ) -> AsyncGenerator[GenerationChunk]:
        raise RuntimeError("Simulated stream response failure")
        return
        yield 
    
    monkeypatch.setattr(RunnerSupervisor, 'stream_response', mock_stream_response)

    task: Task = chat_completion_task(INSTANCE_1_ID, TASK_1_ID)
    await global_events.append_events(
        [
            InstanceCreated(instance=instance_value),
            TaskCreated(task_id=task.task_id, task=task)
        ], 
        origin=MASTER_NODE_ID
    )

    await until_event_with_timeout(global_events, InstanceDeleted)


    events = await global_events.get_events_since(0)

    assert len([x for x in events if isinstance(x.event, RunnerStatusUpdated) and isinstance(x.event.runner_status, FailedRunnerStatus)]) == 3
    assert len([x for x in events if isinstance(x.event, TaskStateUpdated) and x.event.task_status == TaskStatus.FAILED]) == 3
    assert any([isinstance(x.event, InstanceDeleted) for x in events])

    await global_events.append_events(
        [
            InstanceDeleted(
                instance_id=instance_value.instance_id,
            ),
        ], 
        origin=MASTER_NODE_ID
    )

    await asyncio.sleep(0.3)

async def test_stream_response_failed_once(
    monkeypatch: MonkeyPatch,
    worker_running: Callable[[NodeId], Awaitable[tuple[Worker, AsyncSQLiteEventStorage]]],
    instance: Callable[[InstanceId, NodeId, RunnerId], Instance],
    chat_completion_task: Callable[[InstanceId, TaskId], Task]
):
    failed_already = False
    original_stream_response = RunnerSupervisor.stream_response

    async def mock_stream_response(
        self: RunnerSupervisor,
        task: Task,
        request_started_callback: Callable[..., CoroutineType[Any, Any, None]] | None = None,
    ) -> AsyncGenerator[GenerationChunk]:
        nonlocal failed_already
        if not failed_already:
            failed_already = True
            raise RuntimeError("Simulated stream response failure")
        else:
            async for event in original_stream_response(self, task, request_started_callback):
                yield event
            return
    
    monkeypatch.setattr(RunnerSupervisor, 'stream_response', mock_stream_response)
    
    worker, global_events = await worker_running(NODE_A)

    instance_value: Instance = instance(INSTANCE_1_ID, NODE_A, RUNNER_1_ID)
    instance_value.instance_type = InstanceStatus.ACTIVE

    task: Task = chat_completion_task(INSTANCE_1_ID, TASK_1_ID)
    await global_events.append_events(
        [
            InstanceCreated(instance=instance_value),
            TaskCreated(task_id=task.task_id, task=task)
        ], 
        origin=MASTER_NODE_ID
    )

    await until_event_with_timeout(global_events, ChunkGenerated, 1, condition=lambda x: isinstance(x.chunk, TokenChunk) and x.chunk.finish_reason is not None)
    
    # TODO: The ideal with this test is if we had some tooling to scroll through the state, and say
    # 'asser that there was a time that the error_type, error_message was not none and the failure count was nonzero'

    # as we reset the failures back to zero when we have a successful inference.
    assert len(worker.assigned_runners[RUNNER_1_ID].failures) == 0 
    assert worker.state.tasks[TASK_1_ID].error_type is None
    assert worker.state.tasks[TASK_1_ID].error_message is None

    events = await global_events.get_events_since(0)
    assert len([x for x in events if isinstance(x.event, RunnerStatusUpdated) and isinstance(x.event.runner_status, FailedRunnerStatus)]) == 1
    assert len([x for x in events if isinstance(x.event, TaskStateUpdated) and x.event.task_status == TaskStatus.FAILED]) == 1

    response_string = ''
    events = await global_events.get_events_since(0)

    seen_task_started, seen_task_finished = False, False
    for wrapped_event in events:
        event = wrapped_event.event
        if isinstance(event, TaskStateUpdated):
            if event.task_status == TaskStatus.RUNNING:
                seen_task_started = True
            if event.task_status == TaskStatus.COMPLETE:
                seen_task_finished = True

        if isinstance(event, ChunkGenerated):
            assert isinstance(event.chunk, TokenChunk)
            response_string += event.chunk.text

    assert 'queen' in response_string.lower()
    assert seen_task_started
    assert seen_task_finished

    await global_events.append_events(
        [
            InstanceDeleted(
                instance_id=instance_value.instance_id,
            ),
        ], 
        origin=MASTER_NODE_ID
    )

    await asyncio.sleep(0.3)


async def test_stream_response_timeout(
    worker_running: Callable[[NodeId], Awaitable[tuple[Worker, AsyncSQLiteEventStorage]]],
    instance: Callable[[InstanceId, NodeId, RunnerId], Instance],
    chat_completion_task: Callable[[InstanceId, TaskId], Task]
):
    _, global_events = await worker_running(NODE_A)

    instance_value: Instance = instance(INSTANCE_1_ID, NODE_A, RUNNER_1_ID)
    instance_value.instance_type = InstanceStatus.ACTIVE

    task: Task = chat_completion_task(INSTANCE_1_ID, TASK_1_ID)
    task.task_params.messages[0].content = 'EXO RUNNER MUST TIMEOUT'
    await global_events.append_events(
        [
            InstanceCreated(instance=instance_value),
            TaskCreated(task_id=task.task_id, task=task)
        ], 
        origin=MASTER_NODE_ID
    )

    await until_event_with_timeout(global_events, TaskFailed, multiplicity=3)

    events = await global_events.get_events_since(0)
    print(events)
    assert len([x for x in events if isinstance(x.event, RunnerStatusUpdated) and isinstance(x.event.runner_status, FailedRunnerStatus)]) == 3
    assert len([x for x in events if isinstance(x.event, TaskStateUpdated) and x.event.task_status == TaskStatus.FAILED]) == 3
    assert len([x for x in events if isinstance(x.event, TaskFailed) and 'timeouterror' in x.event.error_type.lower()]) == 3

    await global_events.append_events(
        [
            InstanceDeleted(
                instance_id=instance_value.instance_id,
            ),
        ], 
        origin=MASTER_NODE_ID
    )

    await asyncio.sleep(0.3)