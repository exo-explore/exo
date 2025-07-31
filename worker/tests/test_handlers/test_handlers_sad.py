## Tests for worker state handlers

from typing import Callable

import pytest

from shared.types.events import (
    RunnerStatusUpdated,
    TaskFailed,
    TaskStateUpdated,
)
from shared.types.tasks import ChatCompletionTask, TaskStatus
from shared.types.worker.instances import Instance
from shared.types.worker.ops import (
    ExecuteTaskOp,
)
from shared.types.worker.runners import (
    FailedRunnerStatus,
    RunningRunnerStatus,
)
from worker.main import Worker
from worker.tests.constants import RUNNER_1_ID
from worker.tests.test_handlers.utils import read_events_op


@pytest.mark.asyncio
async def test_execute_task_fails(
    worker_with_running_runner: tuple[Worker, Instance], 
    chat_completion_task: Callable[[], ChatCompletionTask]):
    worker, _ = worker_with_running_runner

    task = chat_completion_task()
    messages = task.task_params.messages
    messages[0].content = 'Artificial prompt: EXO RUNNER MUST FAIL'

    execute_task_op = ExecuteTaskOp(
        runner_id=RUNNER_1_ID,
        task=task
    )

    events = await read_events_op(worker, execute_task_op)

    assert len(events) == 5

    print(events)    

    assert isinstance(events[0], RunnerStatusUpdated)
    assert isinstance(events[0].runner_status, RunningRunnerStatus) # It tried to start.

    assert isinstance(events[1], TaskStateUpdated)
    assert events[1].task_status == TaskStatus.RUNNING # It tried to start.

    assert isinstance(events[2], TaskStateUpdated)
    assert events[2].task_status == TaskStatus.FAILED # Task marked as failed. 

    assert isinstance(events[3], TaskFailed)

    assert isinstance(events[4], RunnerStatusUpdated)
    assert isinstance(events[4].runner_status, FailedRunnerStatus) # It should have failed.

# TODO: Much more to do here!