from collections.abc import Mapping
from typing import Sequence

from pydantic import BaseModel

from shared.types.common import NodeId
from shared.types.events.common import InstanceStateEventTypes, State, TaskEventTypes
from shared.types.tasks.common import Task, TaskId, TaskType
from shared.types.worker.common import InstanceId
from shared.types.worker.instances import BaseInstance


class KnownInstances(State[InstanceStateEventTypes]):
    instances: Mapping[InstanceId, BaseInstance]


class Tasks(State[TaskEventTypes]):
    tasks: Mapping[TaskId, Task[TaskType]]


class SharedState(BaseModel):
    node_id: NodeId
    known_instances: KnownInstances
    compute_tasks: Tasks

    def get_node_id(self) -> NodeId: ...

    def get_tasks_by_instance(
        self, instance_id: InstanceId
    ) -> Sequence[Task[TaskType]]: ...
