import asyncio
from logging import Logger
from typing import Callable

import pytest

from exo.shared.types.common import Host
from exo.shared.types.tasks import Task, TaskId
from exo.shared.types.worker.common import InstanceId, RunnerError
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.runner.runner_supervisor import RunnerSupervisor
from exo.worker.tests.constants import INSTANCE_1_ID, TASK_1_ID


@pytest.mark.asyncio
async def test_supervisor_instantiation_exception(
    pipeline_shard_meta: Callable[..., PipelineShardMetadata],
    hosts: Callable[..., list[Host]],
    logger: Logger,
):
    """Test that asking for the capital of France returns 'Paris' in the response"""
    model_shard_meta = pipeline_shard_meta(1, 0)
    model_shard_meta.immediate_exception = True

    with pytest.raises(RunnerError):
        _ = await RunnerSupervisor.create(
            model_shard_meta=model_shard_meta,
            hosts=hosts(1, offset=10),
            logger=logger,
        )


@pytest.mark.asyncio
async def test_supervisor_instantiation_timeout(
    pipeline_shard_meta: Callable[..., PipelineShardMetadata],
    hosts: Callable[..., list[Host]],
    logger: Logger,
):
    """Test that asking for the capital of France returns 'Paris' in the response"""
    model_shard_meta = pipeline_shard_meta(1, 0)
    model_shard_meta.should_timeout = 10  # timeout after 10s

    with pytest.raises(asyncio.TimeoutError):
        _ = await RunnerSupervisor.create(
            model_shard_meta=model_shard_meta,
            hosts=hosts(1, offset=10),
            logger=logger,
        )


@pytest.mark.asyncio
async def test_supervisor_inference_exception(
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
    task.task_params.messages[0].content = "EXO RUNNER MUST FAIL"
    with pytest.raises(RunnerError):
        async for _ in supervisor.stream_response(task):
            pass


@pytest.mark.asyncio
async def test_supervisor_inference_timeout(
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
    task.task_params.messages[0].content = "EXO RUNNER MUST TIMEOUT"
    with pytest.raises(asyncio.TimeoutError):
        async for _ in supervisor.stream_response(task):
            pass

    await asyncio.sleep(0.1)
