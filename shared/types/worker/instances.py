from collections.abc import Mapping

from pydantic import BaseModel

from shared.types.worker.common import InstanceId
from shared.types.worker.runners import (
    RunnerId,
    RunnerPlacement,
    RunnerState,
    RunnerStateType,
)


class InstanceBase(BaseModel):
    instance_id: InstanceId


class InstanceData(BaseModel):
    runner_placements: RunnerPlacement
    runner_states: Mapping[RunnerId, RunnerState[RunnerStateType]]


class Instance(InstanceBase):
    instance_data: InstanceData
