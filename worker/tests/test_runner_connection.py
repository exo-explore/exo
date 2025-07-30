import asyncio
import os
from logging import Logger
from typing import Callable, Final

import pytest

from shared.db.sqlite.event_log_manager import EventLogConfig, EventLogManager
from shared.types.common import Host, NodeId
from shared.types.events import InstanceCreated, InstanceDeleted
from shared.types.models import ModelId
from shared.types.tasks import Task
from shared.types.worker.common import InstanceId, RunnerId
from shared.types.worker.instances import Instance, InstanceStatus, ShardAssignments
from shared.types.worker.runners import FailedRunnerStatus
from shared.types.worker.shards import PipelineShardMetadata
from worker.download.shard_downloader import NoopShardDownloader
from worker.main import Worker

MASTER_NODE_ID = NodeId("ffffffff-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
NODE_A: Final[NodeId] = NodeId("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
NODE_B: Final[NodeId] = NodeId("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")

RUNNER_1_ID: Final[RunnerId] = RunnerId("11111111-1111-4111-8111-111111111111")
INSTANCE_1_ID: Final[InstanceId] = InstanceId("22222222-2222-4222-8222-222222222222")
RUNNER_2_ID: Final[RunnerId] = RunnerId("33333333-3333-4333-8333-333333333333")
INSTANCE_2_ID: Final[InstanceId] = InstanceId("44444444-4444-4444-8444-444444444444")
MODEL_A_ID: Final[ModelId] = 'mlx-community/Llama-3.2-1B-Instruct-4bit'
MODEL_B_ID: Final[ModelId] = 'mlx-community/Llama-3.2-1B-Instruct-4bit'
TASK_1_ID: Final = "55555555-5555-4555-8555-555555555555"
TASK_2_ID: Final = "66666666-6666-4666-8666-666666666666"

@pytest.fixture
def user_message() -> str:
    return "What is the capital of Japan?"

@pytest.mark.skipif(
    os.environ.get("DETAILED", "").lower() != "true",
    reason="This test only runs when ENABLE_SPINUP_TIMEOUT_TEST=true environment variable is set"
)
async def check_runner_connection(
    logger: Logger,
    pipeline_shard_meta: Callable[[int, int], PipelineShardMetadata],
    hosts: Callable[[int], list[Host]],
    chat_completion_task: Callable[[InstanceId, str], Task],
) -> bool:
    # Track all tasks and workers for cleanup
    tasks: list[asyncio.Task[None]] = []
    workers: list[Worker] = []
    
    try:
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
        workers.append(worker1)
        task1 = asyncio.create_task(worker1.run())
        tasks.append(task1)

        worker2 = Worker(
            NODE_B,
            logger=logger,
            shard_downloader=shard_downloader,
            worker_events=global_events,
            global_events=global_events,
        )
        workers.append(worker2)
        task2 = asyncio.create_task(worker2.run())
        tasks.append(task2)

        model_id = ModelId('mlx-community/Llama-3.2-1B-Instruct-4bit')

        shard_assignments = ShardAssignments(
            model_id=model_id,
            runner_to_shard={
                RUNNER_1_ID: pipeline_shard_meta(2, 0),
                RUNNER_2_ID: pipeline_shard_meta(2, 1)
            },
            node_to_runner={
                NODE_A: RUNNER_1_ID,
                NODE_B: RUNNER_2_ID
            }
        )

        instance = Instance(
            instance_id=INSTANCE_1_ID,
            instance_type=InstanceStatus.ACTIVE,
            shard_assignments=shard_assignments,
            hosts=hosts(2)
        )

        await global_events.append_events(
            [
                InstanceCreated(
                    instance=instance
                ),
            ],
            origin=MASTER_NODE_ID
        )

        from worker.runner.runner_supervisor import RunnerSupervisor

        async def wait_for_runner_supervisor(worker: Worker, timeout: float = 5.0) -> RunnerSupervisor | None:
            end = asyncio.get_event_loop().time() + timeout
            while True:
                assigned_runners = list(worker.assigned_runners.values())
                if assigned_runners:
                    runner = assigned_runners[0].runner
                    if isinstance(runner, RunnerSupervisor):
                        print('breaking because success')
                        return runner
                    if isinstance(assigned_runners[0].status, FailedRunnerStatus):
                        print('breaking because failed')
                        return runner
                if asyncio.get_event_loop().time() > end:
                    raise TimeoutError("RunnerSupervisor was not set within timeout")
                await asyncio.sleep(0.001)

        runner_supervisor = await wait_for_runner_supervisor(worker1, timeout=6.0)
        ret = runner_supervisor is not None and runner_supervisor.healthy

        await global_events.append_events(
            [
                InstanceDeleted(
                    instance_id=instance.instance_id,
                ),
            ],
            origin=MASTER_NODE_ID
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

def test_runner_connection_stress(
    logger: Logger,
    pipeline_shard_meta: Callable[[int, int], PipelineShardMetadata],
    hosts: Callable[[int], list[Host]],
    chat_completion_task: Callable[[InstanceId, str], Task],
) -> None:
    total_runs = 100
    successes = 0
    
    for _ in range(total_runs):
        # Create a fresh event loop for each iteration
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(check_runner_connection(
                logger=logger,
                pipeline_shard_meta=pipeline_shard_meta,
                hosts=hosts,
                chat_completion_task=chat_completion_task,
            ))
            if result:
                successes += 1
        finally:
            # Cancel all running tasks
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            
            # Run the event loop briefly to allow cancellation to complete
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            
            # Close the event loop
            loop.close()
    
    print(f"Runner connection successes: {successes} / {total_runs}")
