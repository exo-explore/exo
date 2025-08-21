import asyncio
import os
from logging import Logger
from typing import Callable

import pytest

# TaskStateUpdated and ChunkGenerated are used in test_worker_integration_utils.py
from exo.shared.db.sqlite.event_log_manager import EventLogConfig, EventLogManager
from exo.shared.models.model_meta import get_model_meta
from exo.shared.types.api import ChatCompletionMessage, ChatCompletionTaskParams
from exo.shared.types.common import Host
from exo.shared.types.events import (
    InstanceCreated,
    InstanceDeleted,
    TaskCreated,
)
from exo.shared.types.models import ModelId, ModelMetadata
from exo.shared.types.tasks import (
    ChatCompletionTask,
    Task,
    TaskId,
    TaskStatus,
    TaskType,
)
from exo.shared.types.worker.common import InstanceId
from exo.shared.types.worker.instances import (
    Instance,
    InstanceStatus,
    ShardAssignments,
)
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.download.shard_downloader import NoopShardDownloader
from exo.worker.main import run
from exo.worker.tests.constants import (
    COMMAND_1_ID,
    COMMAND_2_ID,
    INSTANCE_1_ID,
    MASTER_NODE_ID,
    NODE_A,
    NODE_B,
    RUNNER_1_ID,
    RUNNER_2_ID,
    TASK_1_ID,
    TASK_2_ID,
)
from exo.worker.tests.test_integration.integration_utils import (
    read_streaming_response,
)
from exo.worker.worker import Worker

MODEL_ID = "mlx-community/Llama-3.3-70B-Instruct-4bit"


@pytest.fixture
async def model_meta() -> ModelMetadata:
    return await get_model_meta(MODEL_ID)


def _get_model_size_gb(path: str) -> float:
    """Calculate total size of directory recursively in GB."""
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.isfile(filepath):
                total_size += os.path.getsize(filepath)
    return total_size / (1024**3)  # Convert bytes to GB


@pytest.mark.skipif(
    not (
        os.path.exists(
            os.path.expanduser(
                "~/.exo/models/mlx-community--Llama-3.3-70B-Instruct-4bit/"
            )
        )
        and _get_model_size_gb(
            os.path.expanduser(
                "~/.exo/models/mlx-community--Llama-3.3-70B-Instruct-4bit/"
            )
        )
        > 30
    ),
    reason="This test only runs when model mlx-community/Llama-3.3-70B-Instruct-4bit is downloaded",
)
async def test_2_runner_inference(
    logger: Logger,
    pipeline_shard_meta: Callable[[int, int], PipelineShardMetadata],
    hosts: Callable[[int], list[Host]],
    chat_completion_task: Callable[[InstanceId, TaskId], Task],
):
    event_log_manager = EventLogManager(EventLogConfig(), logger)
    await event_log_manager.initialize()
    shard_downloader = NoopShardDownloader()

    global_events = event_log_manager.global_events
    await global_events.delete_all_events()

    worker1 = Worker(
        NODE_A,
        logger=logger,
        shard_downloader=shard_downloader,
        worker_events=global_events,
        global_events=global_events,
    )
    asyncio.create_task(run(worker1, logger))

    worker2 = Worker(
        NODE_B,
        logger=logger,
        shard_downloader=shard_downloader,
        worker_events=global_events,
        global_events=global_events,
    )
    asyncio.create_task(run(worker2, logger))

    ## Instance
    model_id = ModelId(MODEL_ID)

    shard_assignments = ShardAssignments(
        model_id=model_id,
        runner_to_shard={
            RUNNER_1_ID: pipeline_shard_meta(2, 0),
            RUNNER_2_ID: pipeline_shard_meta(2, 1),
        },
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
    )

    instance = Instance(
        instance_id=INSTANCE_1_ID,
        instance_type=InstanceStatus.ACTIVE,
        shard_assignments=shard_assignments,
        hosts=hosts(2),
    )

    task = chat_completion_task(INSTANCE_1_ID, TASK_1_ID)
    task.task_params.messages[
        0
    ].content = (
        "Can you explain to me how a bubble sort works, speaking as if you are a fairy."
    )
    task.task_params.max_tokens = 1000

    await global_events.append_events(
        [
            InstanceCreated(instance=instance),
            TaskCreated(task_id=task.task_id, task=task),
        ],
        origin=MASTER_NODE_ID,
    )

    (
        seen_task_started,
        seen_task_finished,
        response_string,
    ) = await read_streaming_response(global_events)

    assert seen_task_started
    assert seen_task_finished
    assert "swap" in response_string.lower()

    idx = await global_events.get_last_idx()
    await asyncio.sleep(1.0)
    events = await global_events.get_events_since(idx)
    assert len(events) == 0

    await global_events.append_events(
        [
            InstanceDeleted(
                instance_id=instance.instance_id,
            ),
        ],
        origin=MASTER_NODE_ID,
    )

    await asyncio.sleep(2.0)


@pytest.mark.skipif(
    not (
        os.path.exists(
            os.path.expanduser(
                "~/.exo/models/mlx-community--Llama-3.3-70B-Instruct-4bit/"
            )
        )
        and _get_model_size_gb(
            os.path.expanduser(
                "~/.exo/models/mlx-community--Llama-3.3-70B-Instruct-4bit/"
            )
        )
        > 30
    ),
    reason="This test only runs when model mlx-community/Llama-3.3-70B-Instruct-4bit is downloaded",
)
async def test_parallel_inference(
    logger: Logger,
    pipeline_shard_meta: Callable[[int, int], PipelineShardMetadata],
    hosts: Callable[[int], list[Host]],
    chat_completion_task: Callable[[InstanceId, TaskId], Task],
):
    event_log_manager = EventLogManager(EventLogConfig(), logger)
    await event_log_manager.initialize()
    shard_downloader = NoopShardDownloader()

    global_events = event_log_manager.global_events
    await global_events.delete_all_events()

    worker1 = Worker(
        NODE_A,
        logger=logger,
        shard_downloader=shard_downloader,
        worker_events=global_events,
        global_events=global_events,
    )
    asyncio.create_task(run(worker1, logger))

    worker2 = Worker(
        NODE_B,
        logger=logger,
        shard_downloader=shard_downloader,
        worker_events=global_events,
        global_events=global_events,
    )
    asyncio.create_task(run(worker2, logger))

    ## Instance
    model_id = ModelId(MODEL_ID)

    shard_assignments = ShardAssignments(
        model_id=model_id,
        runner_to_shard={
            RUNNER_1_ID: pipeline_shard_meta(2, 0),
            RUNNER_2_ID: pipeline_shard_meta(2, 1),
        },
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
    )

    instance = Instance(
        instance_id=INSTANCE_1_ID,
        instance_type=InstanceStatus.ACTIVE,
        shard_assignments=shard_assignments,
        hosts=hosts(2),
    )

    completion_create_params_1 = ChatCompletionTaskParams(
        model="gpt-4",
        messages=[
            ChatCompletionMessage(
                role="user", content='Tell me a haiku that uses the word "pond".'
            )
        ],
        stream=True,
        max_tokens=1000,
    )
    task1 = ChatCompletionTask(
        task_id=TASK_1_ID,
        command_id=COMMAND_1_ID,
        instance_id=INSTANCE_1_ID,
        task_type=TaskType.CHAT_COMPLETION,
        task_status=TaskStatus.PENDING,
        task_params=completion_create_params_1,
    )

    completion_create_params_2 = ChatCompletionTaskParams(
        model="gpt-4",
        messages=[
            ChatCompletionMessage(
                role="user", content='Tell me a haiku that uses the word "tree".'
            )
        ],
        stream=True,
        max_tokens=1000,
    )
    task2 = ChatCompletionTask(
        task_id=TASK_2_ID,
        command_id=COMMAND_2_ID,
        instance_id=INSTANCE_1_ID,
        task_type=TaskType.CHAT_COMPLETION,
        task_status=TaskStatus.PENDING,
        task_params=completion_create_params_2,
    )

    await global_events.append_events(
        [
            InstanceCreated(instance=instance),
            TaskCreated(task_id=task1.task_id, task=task1),
            TaskCreated(task_id=task2.task_id, task=task2),
        ],
        origin=MASTER_NODE_ID,
    )

    (
        seen_task_started_1,
        seen_task_finished_1,
        response_string_1,
    ) = await read_streaming_response(global_events)

    incomplete_task = (
        TASK_2_ID
        if worker1.state.tasks[TASK_1_ID].task_status == TaskStatus.COMPLETE
        else TASK_2_ID
    )
    (
        seen_task_started_2,
        seen_task_finished_2,
        response_string_2,
    ) = await read_streaming_response(global_events, filter_task=incomplete_task)

    assert seen_task_started_1
    assert seen_task_finished_1
    assert seen_task_started_2
    assert seen_task_finished_2

    print(response_string_1)
    print(response_string_2)

    assert ("pond" in response_string_1.lower()) ^ (
        "pond" in response_string_2.lower()
    ), "'pond' must appear in exactly one response"
    assert ("tree" in response_string_1.lower()) ^ (
        "tree" in response_string_2.lower()
    ), "'tree' must appear in exactly one response"

    idx = await global_events.get_last_idx()
    await asyncio.sleep(1.0)
    events = await global_events.get_events_since(idx)
    assert len(events) == 0

    await global_events.append_events(
        [
            InstanceDeleted(
                instance_id=instance.instance_id,
            ),
        ],
        origin=MASTER_NODE_ID,
    )

    await asyncio.sleep(2.0)
