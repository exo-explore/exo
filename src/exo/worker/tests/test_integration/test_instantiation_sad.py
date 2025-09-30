import asyncio
from typing import Callable

from anyio import create_task_group

# TaskStateUpdated and ChunkGenerated are used in test_worker_integration_utils.py
from exo.shared.types.common import NodeId

# TaskStateUpdated and ChunkGenerated are used in test_worker_integration_utils.py
from exo.shared.types.events import (
    InstanceCreated,
    InstanceDeleted,
    RunnerStatusUpdated,
)
from exo.shared.types.worker.common import InstanceId, RunnerId
from exo.shared.types.worker.instances import (
    Instance,
    InstanceStatus,
)
from exo.shared.types.worker.runners import (
    FailedRunnerStatus,
)
from exo.worker.main import Worker
from exo.worker.tests.constants import (
    INSTANCE_1_ID,
    MASTER_NODE_ID,
    NODE_A,
    RUNNER_1_ID,
)
from exo.worker.tests.worker_management import WorkerMailbox, until_event_with_timeout


async def test_runner_spinup_exception(
    instance: Callable[[InstanceId, NodeId, RunnerId], Instance],
    worker_and_mailbox: tuple[Worker, WorkerMailbox],
):
    worker, global_events = worker_and_mailbox
    async with create_task_group() as tg:
        tg.start_soon(worker.run)
        instance_value: Instance = instance(INSTANCE_1_ID, NODE_A, RUNNER_1_ID)
        instance_value.instance_type = InstanceStatus.ACTIVE
        instance_value.shard_assignments.runner_to_shard[
            RUNNER_1_ID
        ].immediate_exception = True

        await global_events.append_events(
            [InstanceCreated(instance=instance_value)], origin=MASTER_NODE_ID
        )

        await asyncio.sleep(10.0)

        # Ensure the correct events have been emitted
        events = global_events.collect()

        assert (
            len(
                [
                    x
                    for x in events
                    if isinstance(x.tagged_event.c, RunnerStatusUpdated)
                    and isinstance(x.tagged_event.c.runner_status, FailedRunnerStatus)
                ]
            )
            == 3
        )
        assert any([isinstance(x.tagged_event.c, InstanceDeleted) for x in events])
        worker.shutdown()


async def test_runner_spinup_timeout(
    instance: Callable[[InstanceId, NodeId, RunnerId], Instance],
    worker_and_mailbox: tuple[Worker, WorkerMailbox],
):
    worker, global_events = worker_and_mailbox
    async with create_task_group() as tg:
        tg.start_soon(worker.run)
        instance_value: Instance = instance(INSTANCE_1_ID, NODE_A, RUNNER_1_ID)
        instance_value.instance_type = InstanceStatus.ACTIVE
        instance_value.shard_assignments.runner_to_shard[
            RUNNER_1_ID
        ].should_timeout = 10

        await global_events.append_events(
            [InstanceCreated(instance=instance_value)], origin=MASTER_NODE_ID
        )

        await until_event_with_timeout(
            global_events,
            RunnerStatusUpdated,
            multiplicity=3,
            condition=lambda x: isinstance(x.runner_status, FailedRunnerStatus),
        )

        # Ensure the correct events have been emitted
        events = global_events.collect()

        assert (
            len(
                [
                    x
                    for x in events
                    if isinstance(x.tagged_event.c, RunnerStatusUpdated)
                    and isinstance(x.tagged_event.c.runner_status, FailedRunnerStatus)
                ]
            )
            == 3
        )
        assert any([isinstance(x.tagged_event.c, InstanceDeleted) for x in events])
        worker.shutdown()
