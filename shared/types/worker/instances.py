from enum import Enum

from pydantic import BaseModel

from shared.types.worker.common import InstanceId
from shared.types.worker.mlx import Host
from shared.types.worker.runners import (
    ShardAssignments,
)


class InstanceStatus(str, Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"

class Instance(BaseModel):
    instance_id: InstanceId
    instance_type: InstanceStatus
    shard_assignments: ShardAssignments
    hosts: list[Host]
