from pydantic import BaseModel

from shared.types.common import NodeId
from shared.types.worker.common import InstanceId
from shared.types.worker.shards import ShardPlacement


class SharedState(BaseModel):
    node_id: NodeId
    compute_instances: dict[InstanceId, ShardPlacement]
