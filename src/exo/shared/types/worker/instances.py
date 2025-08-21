from enum import Enum

from pydantic import BaseModel

from exo.shared.types.common import Host
from exo.shared.types.worker.common import InstanceId
from exo.shared.types.worker.runners import (
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
