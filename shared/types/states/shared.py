from collections.abc import Mapping

from pydantic import BaseModel

from shared.types.common import NodeId
from shared.types.tasks.common import Task, TaskId, TaskType
from shared.types.worker.common import InstanceId
from shared.types.worker.instances import InstanceData


class SharedState(BaseModel):
    node_id: NodeId
    compute_instances: Mapping[InstanceId, InstanceData]
    compute_tasks: dict[TaskId, Task[TaskType]]
