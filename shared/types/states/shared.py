from collections.abc import Mapping
from typing import Literal, Sequence

from pydantic import BaseModel

from shared.types.common import NodeId
from shared.types.events.common import EventCategoryEnum, State
from shared.types.tasks.common import (
    Task,
    TaskId,
    TaskSagaEntry,
    TaskStatusType,
    TaskType,
)
from shared.types.worker.common import InstanceId
from shared.types.worker.instances import BaseInstance
from shared.types.worker.runners import RunnerId, RunnerStatus


class Instances(State[EventCategoryEnum.MutatesInstanceState]):
    event_category: Literal[EventCategoryEnum.MutatesInstanceState] = (
        EventCategoryEnum.MutatesInstanceState
    )
    instances: Mapping[InstanceId, BaseInstance] = {}


class Tasks(State[EventCategoryEnum.MutatesTaskState]):
    event_category: Literal[EventCategoryEnum.MutatesTaskState] = (
        EventCategoryEnum.MutatesTaskState
    )
    tasks: Mapping[TaskId, Task[TaskType, TaskStatusType]] = {}


class TaskSagas(State[EventCategoryEnum.MutatesTaskSagaState]):
    event_category: Literal[EventCategoryEnum.MutatesTaskSagaState] = (
        EventCategoryEnum.MutatesTaskSagaState
    )
    task_sagas: Mapping[TaskId, Sequence[TaskSagaEntry]] = {}


class Runners(State[EventCategoryEnum.MutatesRunnerStatus]):
    event_category: Literal[EventCategoryEnum.MutatesRunnerStatus] = (
        EventCategoryEnum.MutatesRunnerStatus
    )
    runner_statuses: Mapping[RunnerId, RunnerStatus] = {}


class SharedState(BaseModel):
    instances: Instances = Instances()
    runners: Runners = Runners()
    tasks: Tasks = Tasks()
    task_sagas: TaskSagas = TaskSagas()

    def get_node_id(self) -> NodeId: ...

    def get_tasks_by_instance(
        self, instance_id: InstanceId
    ) -> Sequence[Task[TaskType, TaskStatusType]]: ...
