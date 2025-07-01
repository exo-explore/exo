from collections.abc import Mapping

from pydantic import BaseModel

from shared.types.worker.common import InstanceId
from shared.types.worker.runners import (
    RunnerId,
    RunnerPlacement,
    RunnerState,
    RunnerStateType,
)


class InstanceState(BaseModel):
    runner_states: Mapping[RunnerId, RunnerState[RunnerStateType]]


class InstanceData(BaseModel):
    runner_placements: RunnerPlacement


class Instance(BaseModel):
    instance_id: InstanceId
    instance_data: InstanceData
    instance_state: InstanceState
