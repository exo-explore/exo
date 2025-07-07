from collections.abc import Mapping
from enum import Enum

from pydantic import BaseModel

from shared.types.worker.common import InstanceId
from shared.types.worker.runners import (
    RunnerId,
    RunnerState,
    RunnerStateType,
    ShardAssignments,
)


class InstanceStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"


class InstanceState(BaseModel):
    runner_states: Mapping[RunnerId, RunnerState[RunnerStateType]]


class InstanceData(BaseModel):
    shard_assignments: ShardAssignments


class BaseInstance(BaseModel):
    instance_data: InstanceData
    instance_state: InstanceState
    instance_status: InstanceStatus


class Instance(BaseInstance):
    instance_id: InstanceId
