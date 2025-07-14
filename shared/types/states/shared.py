from collections.abc import Mapping
from typing import Literal, Sequence

from pydantic import BaseModel

from shared.types.common import NodeId
from shared.types.events.common import EventCategories, State
from shared.types.tasks.common import Task, TaskId, TaskStatusType, TaskType
from shared.types.worker.common import InstanceId
from shared.types.worker.instances import BaseInstance


class KnownInstances(State[EventCategories.InstanceStateEventTypes]):
    event_category: Literal[EventCategories.InstanceStateEventTypes] = (
        EventCategories.InstanceStateEventTypes
    )
    instances: Mapping[InstanceId, BaseInstance] = {}


class Tasks(State[EventCategories.TaskEventTypes]):
    event_category: Literal[EventCategories.TaskEventTypes] = (
        EventCategories.TaskEventTypes
    )
    tasks: Mapping[TaskId, Task[TaskType, TaskStatusType]] = {}


class SharedState(BaseModel):
    known_instances: KnownInstances = KnownInstances()
    compute_tasks: Tasks = Tasks()

    def get_node_id(self) -> NodeId: ...

    def get_tasks_by_instance(
        self, instance_id: InstanceId
    ) -> Sequence[Task[TaskType, TaskStatusType]]: ...
