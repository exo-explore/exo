from enum import Enum

from exo.shared.types.common import Host
from exo.shared.types.worker.common import InstanceId
from exo.shared.types.worker.runners import (
    ShardAssignments,
)
from exo.utils.pydantic_ext import CamelCaseModel


class InstanceStatus(str, Enum):
    Active = "Active"
    Inactive = "Inactive"


class Instance(CamelCaseModel):
    instance_id: InstanceId
    instance_type: InstanceStatus
    shard_assignments: ShardAssignments
    hosts: list[Host] | None = None
    mlx_ibv_devices: list[list[str | None]] | None = None
    mlx_ibv_coordinator: str | None = None
