## Tests for worker state handlers

import asyncio
from typing import Callable

import pytest

from exo.shared.types.tasks import ChatCompletionTask
from exo.shared.types.worker.common import RunnerError
from exo.shared.types.worker.instances import Instance
from exo.shared.types.worker.ops import (
    ExecuteTaskOp,
    RunnerUpOp,
)
from exo.worker.main import Worker
from exo.worker.tests.constants import RUNNER_1_ID
from exo.worker.tests.test_handlers.utils import read_events_op


@pytest.mark.asyncio
async def test_runner_up_fails(
    worker_with_assigned_runner: tuple[Worker, Instance], 
    chat_completion_task: Callable[[], ChatCompletionTask]):
    worker, _ = worker_with_assigned_runner
    worker.assigned_runners[RUNNER_1_ID].shard_metadata.immediate_exception = True

    runner_up_op = RunnerUpOp(runner_id=RUNNER_1_ID)

    with pytest.raises(RunnerError):
        await read_events_op(worker, runner_up_op)

@pytest.mark.asyncio
async def test_runner_up_timeouts(
    worker_with_assigned_runner: tuple[Worker, Instance], 
    chat_completion_task: Callable[[], ChatCompletionTask]):
    worker, _ = worker_with_assigned_runner
    worker.assigned_runners[RUNNER_1_ID].shard_metadata.should_timeout = 10

    runner_up_op = RunnerUpOp(runner_id=RUNNER_1_ID)

    with pytest.raises(asyncio.TimeoutError):
        await read_events_op(worker, runner_up_op)

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

    with pytest.raises(RunnerError):
        await read_events_op(worker, execute_task_op)

@pytest.mark.asyncio
async def test_execute_task_timeouts(
    worker_with_running_runner: tuple[Worker, Instance], 
    chat_completion_task: Callable[[], ChatCompletionTask]):
    worker, _ = worker_with_running_runner

    task = chat_completion_task()
    messages = task.task_params.messages
    messages[0].content = 'Artificial prompt: EXO RUNNER MUST TIMEOUT'

    execute_task_op = ExecuteTaskOp(
        runner_id=RUNNER_1_ID,
        task=task
    )

    with pytest.raises(asyncio.TimeoutError):
        await read_events_op(worker, execute_task_op)


# TODO: Much more to do here!
# runner assigned download stuff