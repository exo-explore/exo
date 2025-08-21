from logging import Logger
from typing import Callable

import pytest

from exo.shared.types.common import Host
from exo.shared.types.tasks import (
    Task,
    TaskId,
)
from exo.shared.types.worker.common import InstanceId, RunnerError
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.runner.runner_supervisor import RunnerSupervisor
from exo.worker.tests.constants import INSTANCE_1_ID, TASK_1_ID


@pytest.fixture
def user_message():
    """Override the default message to ask about France's capital"""
    return "What is the capital of France?"


@pytest.mark.asyncio
@pytest.mark.skip(
    reason="Must run `sudo sysctl -w iogpu.wired_limit_mb=` and `sudo sysctl -w iogpu.wired_lwm_mb=` before running this test."
)
async def test_supervisor_catches_oom(
    pipeline_shard_meta: Callable[..., PipelineShardMetadata],
    hosts: Callable[..., list[Host]],
    chat_completion_task: Callable[[InstanceId, TaskId], Task],
    logger: Logger,
):
    """Test that asking for the capital of France returns 'Paris' in the response"""
    model_shard_meta = pipeline_shard_meta(1, 0)

    supervisor = await RunnerSupervisor.create(
        model_shard_meta=model_shard_meta,
        hosts=hosts(1, offset=10),
        logger=logger,
    )

    task = chat_completion_task(INSTANCE_1_ID, TASK_1_ID)
    task.task_params.messages[0].content = "EXO RUNNER MUST OOM"
    with pytest.raises(RunnerError) as exc_info:
        async for _ in supervisor.stream_response(task):
            pass

    error = exc_info.value
    assert "memory" in error.error_message.lower()

    await supervisor.astop()
