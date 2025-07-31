## Tests for worker state handlers

import os
from typing import Callable

import pytest

from shared.types.events import (
    Event,
)
from shared.types.events._events import RunnerStatusUpdated
from shared.types.tasks import Task, TaskId
from shared.types.worker.instances import Instance, InstanceId
from shared.types.worker.ops import (
    RunnerUpOp,
)
from shared.types.worker.runners import FailedRunnerStatus
from worker.main import Worker
from worker.tests.constants import RUNNER_1_ID

# To enable this test, run pytest with: ENABLE_SPINUP_TIMEOUT_TEST=true pytest

@pytest.mark.skipif(
    os.environ.get("DETAILED", "").lower() != "true",
    reason="This test only runs when ENABLE_SPINUP_TIMEOUT_TEST=true environment variable is set"
)
@pytest.mark.asyncio
async def test_runner_up_op_timeout(
    worker_with_assigned_runner: tuple[Worker, Instance], 
    chat_completion_task: Callable[[InstanceId, TaskId], Task], 
    monkeypatch: pytest.MonkeyPatch
    ):
    worker, _ = worker_with_assigned_runner

    runner_up_op = RunnerUpOp(runner_id=RUNNER_1_ID)

    # _execute_runner_up_op should throw a TimeoutError with a short timeout
    events: list[Event] = []
    async for event in worker._execute_runner_up_op(runner_up_op, initialize_timeout=0.2):  # type: ignore[misc]
        events.append(event)

    assert isinstance(events[-1], RunnerStatusUpdated)
    assert isinstance(events[-1].runner_status, FailedRunnerStatus)
    assert events[-1].runner_status.error_message is not None
    assert 'timeout' in events[-1].runner_status.error_message.lower()

    del worker.assigned_runners[list(worker.assigned_runners.keys())[0]]

