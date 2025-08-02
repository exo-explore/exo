from logging import Logger
from typing import Callable

import pytest

from shared.types.common import Host
from shared.types.tasks import (
    Task,
    TaskId,
)
from shared.types.worker.common import InstanceId, RunnerError
from shared.types.worker.shards import PipelineShardMetadata
from worker.runner.runner_supervisor import RunnerSupervisor
from worker.tests.constants import INSTANCE_1_ID, TASK_1_ID


@pytest.fixture
def user_message():
    """Override the default message to ask about France's capital"""
    return "What is the capital of France?"


@pytest.mark.asyncio
async def test_supervisor_single_node_response(
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
    task.task_params.messages[0].content = 'EXO RUNNER MUST OOM'
    with pytest.raises(RunnerError):
            async for _ in supervisor.stream_response(task):
                pass

    await supervisor.astop()
