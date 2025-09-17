from logging import Logger
from multiprocessing import Process
from typing import Callable

import psutil
import pytest

from exo.shared.logging import logger_test_install
from exo.shared.models.model_meta import get_model_meta
from exo.shared.types.common import Host
from exo.shared.types.models import ModelMetadata
from exo.shared.types.tasks import Task, TaskId
from exo.shared.types.worker.common import InstanceId, RunnerError
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.runner.runner_supervisor import RunnerSupervisor
from exo.worker.tests.constants import INSTANCE_1_ID, TASK_1_ID


def get_memory_mb(process: Process) -> float:
    """
    Returns the resident set size (RSS) memory usage in MiB for the given process.
    """
    ps = psutil.Process(process.pid)
    rss_bytes: int = ps.memory_info().rss  # type: ignore[attr-defined]
    return rss_bytes / (1024 * 1024)


@pytest.fixture
async def model_meta() -> ModelMetadata:
    return await get_model_meta("mlx-community/Llama-3.3-70B-Instruct-4bit")


@pytest.mark.asyncio
async def test_supervisor_inference_exception(
    pipeline_shard_meta: Callable[..., PipelineShardMetadata],
    hosts: Callable[..., list[Host]],
    chat_completion_task: Callable[[InstanceId, TaskId], Task],
    logger: Logger,
):
    logger_test_install(logger)
    model_shard_meta = pipeline_shard_meta(1, 0)

    supervisor = await RunnerSupervisor.create(
        model_shard_meta=model_shard_meta,
        hosts=hosts(1, offset=10),
    )

    process: Process = supervisor.runner_process
    memory = get_memory_mb(process)
    assert memory > 30 * 100

    task = chat_completion_task(INSTANCE_1_ID, TASK_1_ID)
    task.task_params.messages[0].content = "EXO RUNNER MUST FAIL"
    with pytest.raises(RunnerError):
        async for _ in supervisor.stream_response(task):
            pass

    await supervisor.astop()

    available_memory_bytes: int = psutil.virtual_memory().available
    print(available_memory_bytes // (2**30))
    assert available_memory_bytes > 30 * 2**30
