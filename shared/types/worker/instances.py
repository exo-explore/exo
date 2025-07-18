from enum import Enum

from pydantic import BaseModel

from shared.types.worker.common import InstanceId
from shared.types.worker.mlx import Host
from shared.types.worker.runners import (
    ShardAssignments,
)


class TypeOfInstance(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"


class InstanceParams(BaseModel):
    shard_assignments: ShardAssignments
    hosts: list[Host]


class BaseInstance(BaseModel):
    instance_params: InstanceParams
    instance_type: TypeOfInstance


class Instance(BaseInstance):
    instance_id: InstanceId
