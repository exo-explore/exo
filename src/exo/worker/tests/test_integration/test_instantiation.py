from logging import Logger
from typing import Callable

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
from exo.worker.tests.constants import (
    INSTANCE_1_ID,
    MASTER_NODE_ID,
    NODE_A,
    RUNNER_1_ID,
)
from exo.worker.tests.test_integration.integration_utils import (
    until_event_with_timeout,
    worker_running,
)


async def test_runner_spinup_timeout(
    instance: Callable[[InstanceId, NodeId, RunnerId], Instance],
    logger: Logger,
):
    async with worker_running(NODE_A, logger) as (_, global_events):
        instance_value: Instance = instance(INSTANCE_1_ID, NODE_A, RUNNER_1_ID)
        instance_value.instance_type = InstanceStatus.ACTIVE
        instance_value.shard_assignments.runner_to_shard[RUNNER_1_ID].should_timeout = 10

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
        events = await global_events.get_events_since(0)

        assert (
            len(
                [
                    x
                    for x in events
                    if isinstance(x.event, RunnerStatusUpdated)
                    and isinstance(x.event.runner_status, FailedRunnerStatus)
                ]
            )
            == 3
        )
        assert any([isinstance(x.event, InstanceDeleted) for x in events])