from copy import deepcopy
from typing import Optional

from pydantic import BaseModel, ConfigDict

from shared.types.common import Host
from shared.types.events import (
    InstanceId,
    RunnerStatusUpdated,
)
from shared.types.worker.common import RunnerId
from shared.types.worker.runners import (
    RunnerStatus,
)
from shared.types.worker.shards import ShardMetadata
from worker.runner.runner_supervisor import RunnerSupervisor


class AssignedRunner(BaseModel):
    runner_id: RunnerId
    instance_id: InstanceId
    shard_metadata: ShardMetadata  # just data
    hosts: list[Host]

    status: RunnerStatus
    failures: list[tuple[float, Exception]] = []
    runner: Optional[RunnerSupervisor]  # set if the runner is 'up'

    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def status_update_event(self) -> RunnerStatusUpdated:
        return RunnerStatusUpdated(
            runner_id=self.runner_id,
            runner_status=deepcopy(self.status),
        )
