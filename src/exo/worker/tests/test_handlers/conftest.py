from logging import Logger
from typing import Callable

import pytest

from exo.shared.db.sqlite.event_log_manager import EventLogConfig, EventLogManager
from exo.shared.types.common import NodeId
from exo.shared.types.worker.common import InstanceId
from exo.shared.types.worker.instances import Instance
from exo.shared.types.worker.ops import (
    AssignRunnerOp,
    RunnerUpOp,
)
from exo.shared.types.worker.runners import RunnerId
from exo.worker.download.shard_downloader import NoopShardDownloader
from exo.worker.tests.constants import INSTANCE_1_ID, NODE_A, RUNNER_1_ID
from exo.worker.worker import Worker


@pytest.fixture
def user_message():
    return "What, according to Douglas Adams, is the meaning of life, the universe and everything?"


@pytest.fixture
async def worker(logger: Logger):
    event_log_manager = EventLogManager(EventLogConfig(), logger)
    shard_downloader = NoopShardDownloader()
    await event_log_manager.initialize()

    return Worker(NODE_A, logger, shard_downloader, worker_events=event_log_manager.global_events, global_events=event_log_manager.global_events)

# TODO: instance_id and runner_id are selectable.
@pytest.fixture
async def worker_with_assigned_runner(worker: Worker, instance: Callable[[InstanceId, NodeId, RunnerId], Instance]):
    """Fixture that provides a worker with an already assigned runner."""
    
    instance_id = INSTANCE_1_ID
    runner_id = RUNNER_1_ID
    instance_obj: Instance = instance(instance_id, worker.node_id, runner_id)
    
    # Assign the runner
    assign_op = AssignRunnerOp(
        runner_id=runner_id,
        shard_metadata=instance_obj.shard_assignments.runner_to_shard[runner_id],
        hosts=instance_obj.hosts,
        instance_id=instance_obj.instance_id,
    )
    
    async for _ in worker.execute_op(assign_op):
        pass
    
    return worker, instance_obj

@pytest.fixture
async def worker_with_running_runner(worker_with_assigned_runner: tuple[Worker, Instance]):
    """Fixture that provides a worker with an already assigned runner."""
    worker, instance_obj = worker_with_assigned_runner

    runner_up_op = RunnerUpOp(runner_id=RUNNER_1_ID)
    async for _ in worker.execute_op(runner_up_op):
        pass

    # Is the runner actually running?
    supervisor = next(iter(worker.assigned_runners.values())).runner
    assert supervisor is not None
    assert supervisor.healthy

    return worker, instance_obj

