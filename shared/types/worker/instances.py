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


class TypeOfInstance(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"


class InstanceParams(BaseModel):
    shard_assignments: ShardAssignments


class BaseInstance(BaseModel):
    instance_params: InstanceParams
    instance_type: TypeOfInstance


class Instance(BaseInstance):
    instance_id: InstanceId


class BaseInstanceSaga(BaseModel):
    runner_states: Mapping[RunnerId, RunnerState[RunnerStateType]]


class InstanceSaga(BaseInstanceSaga):
    instance_id: InstanceId
