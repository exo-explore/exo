import asyncio
import os
from logging import Logger
from typing import Callable

import pytest

from exo.shared.db.sqlite.event_log_manager import EventLogConfig, EventLogManager
from exo.shared.logging import logger_test_install
from exo.shared.types.common import Host
from exo.shared.types.events import InstanceCreated, InstanceDeleted
from exo.shared.types.models import ModelId
from exo.shared.types.worker.instances import Instance, InstanceStatus, ShardAssignments
from exo.shared.types.worker.runners import FailedRunnerStatus
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.download.shard_downloader import NoopShardDownloader
from exo.worker.main import run
from exo.worker.tests.constants import (
    INSTANCE_1_ID,
    MASTER_NODE_ID,
    NODE_A,
    NODE_B,
    RUNNER_1_ID,
    RUNNER_2_ID,
)
from exo.worker.worker import Worker


@pytest.fixture
def user_message() -> str:
    return "What is the capital of Japan?"


@pytest.mark.skipif(
    os.environ.get("DETAILED", "").lower() != "true",
    reason="This test only runs when ENABLE_SPINUP_TIMEOUT_TEST=true environment variable is set",
)
async def check_runner_connection(
    logger: Logger,
    pipeline_shard_meta: Callable[[int, int], PipelineShardMetadata],
    hosts: Callable[[int], list[Host]],
) -> bool:
    logger_test_install(logger)
    # Track all tasks and workers for cleanup
    tasks: list[asyncio.Task[None]] = []
    workers: list[Worker] = []

    try:
        event_log_manager = EventLogManager(EventLogConfig())
        await event_log_manager.initialize()
        shard_downloader = NoopShardDownloader()

        global_events = event_log_manager.global_events
        await global_events.delete_all_events()

        worker1 = Worker(
            NODE_A,
            shard_downloader=shard_downloader,
            worker_events=global_events,
            global_events=global_events,
        )
        workers.append(worker1)
        task1 = asyncio.create_task(run(worker1))
        tasks.append(task1)

        worker2 = Worker(
            NODE_B,
            shard_downloader=shard_downloader,
            worker_events=global_events,
            global_events=global_events,
        )
        workers.append(worker2)
        task2 = asyncio.create_task(run(worker2))
        tasks.append(task2)

        model_id = ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit")

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

        await global_events.append_events(
            [
                InstanceCreated(instance=instance),
            ],
            origin=MASTER_NODE_ID,
        )

        from exo.worker.runner.runner_supervisor import RunnerSupervisor

        async def wait_for_runner_supervisor(
            worker: Worker, timeout: float = 5.0
        ) -> RunnerSupervisor | None:
            end = asyncio.get_event_loop().time() + timeout
            while True:
                assigned_runners = list(worker.assigned_runners.values())
                if assigned_runners:
                    runner = assigned_runners[0].runner
                    if isinstance(runner, RunnerSupervisor):
                        print("breaking because success")
                        return runner
                    if isinstance(assigned_runners[0].status, FailedRunnerStatus):
                        print("breaking because failed")
                        return runner
                if asyncio.get_event_loop().time() > end:
                    raise TimeoutError("RunnerSupervisor was not set within timeout")
                await asyncio.sleep(0.001)

        runner_supervisor = await wait_for_runner_supervisor(worker1, timeout=6.0)
        ret = runner_supervisor is not None and runner_supervisor.runner_process.is_alive()

        await global_events.append_events(
            [
                InstanceDeleted(
                    instance_id=instance.instance_id,
                ),
            ],
            origin=MASTER_NODE_ID,
        )

        await asyncio.sleep(0.5)

        return ret
    finally:
        # Cancel all worker tasks
        for task in tasks:
            task.cancel()

        # Wait for cancellation to complete
        await asyncio.gather(*tasks, return_exceptions=True)


# Check Running status

# # not now.

# def test_runner_connection_stress(
#     logger: Logger,
#     pipeline_shard_meta: Callable[[int, int], PipelineShardMetadata],
#     hosts: Callable[[int], list[Host]],
#     chat_completion_task: Callable[[InstanceId, str], Task],
# ) -> None:
#     total_runs = 100
#     successes = 0
# # not now.

# def test_runner_connection_stress(
#     logger: Logger,
#     pipeline_shard_meta: Callable[[int, int], PipelineShardMetadata],
#     hosts: Callable[[int], list[Host]],
#     chat_completion_task: Callable[[InstanceId, str], Task],
# ) -> None:
#     logger_test_install(logger)
#     total_runs = 100
#     successes = 0

#     for _ in range(total_runs):
#         # Create a fresh event loop for each iteration
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
#     for _ in range(total_runs):
#         # Create a fresh event loop for each iteration
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)

#         try:
#             result = loop.run_until_complete(check_runner_connection(
#                 pipeline_shard_meta=pipeline_shard_meta,
#                 hosts=hosts,
#                 chat_completion_task=chat_completion_task,
#             ))
#             if result:
#                 successes += 1
#         finally:
#             # Cancel all running tasks
#             pending = asyncio.all_tasks(loop)
#             for task in pending:
#                 task.cancel()
#         try:
#             result = loop.run_until_complete(check_runner_connection(
#                 pipeline_shard_meta=pipeline_shard_meta,
#                 hosts=hosts,
#                 chat_completion_task=chat_completion_task,
#             ))
#             if result:
#                 successes += 1
#         finally:
#             # Cancel all running tasks
#             pending = asyncio.all_tasks(loop)
#             for task in pending:
#                 task.cancel()

#             # Run the event loop briefly to allow cancellation to complete
#             loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
#             # Run the event loop briefly to allow cancellation to complete
#             loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

#             # Close the event loop
#             loop.close()
#             # Close the event loop
#             loop.close()

#     print(f"Runner connection successes: {successes} / {total_runs}")
#     print(f"Runner connection successes: {successes} / {total_runs}")
