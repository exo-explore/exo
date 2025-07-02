from __future__ import annotations

from typing import Any, Literal, Tuple

from pydantic import BaseModel

from shared.types.common import NewUUID, NodeId
from shared.types.events.common import (
    Event,
    InstanceEventTypes,
    MLXEventTypes,
    NodeEventTypes,
    StreamingEventTypes,
    TaskEventTypes,
    TimerEventTypes,
)
from shared.types.profiling.common import NodeProfile
from shared.types.tasks.common import (
    TaskData,
    TaskId,
    TaskStatusType,
    TaskType,
    TaskUpdate,
)
from shared.types.worker.common import InstanceId, NodeState
from shared.types.worker.instances import InstanceData
from shared.types.worker.runners import RunnerId, RunnerState, RunnerStateType


class RequestId(NewUUID):
    pass


class TimerId(NewUUID):
    pass


class TimerData(BaseModel):
    timer_id: TimerId


class TaskCreated(Event[TaskEventTypes.TaskCreated]):
    event_type: Literal[TaskEventTypes.TaskCreated] = TaskEventTypes.TaskCreated
    task_id: TaskId
    task_data: TaskData[TaskType]
    task_state: TaskUpdate[Literal[TaskStatusType.Pending]]
    on_instance: InstanceId


class TaskUpdated(Event[TaskEventTypes.TaskUpdated]):
    event_type: Literal[TaskEventTypes.TaskUpdated] = TaskEventTypes.TaskUpdated
    task_id: TaskId
    update_data: TaskUpdate[TaskStatusType]


class TaskDeleted(Event[TaskEventTypes.TaskDeleted]):
    event_type: Literal[TaskEventTypes.TaskDeleted] = TaskEventTypes.TaskDeleted
    task_id: TaskId


class InstanceCreated(Event[InstanceEventTypes.InstanceCreated]):
    event_type: Literal[InstanceEventTypes.InstanceCreated] = (
        InstanceEventTypes.InstanceCreated
    )
    instance_id: InstanceId
    instance_data: InstanceData


class InstanceDeleted(Event[InstanceEventTypes.InstanceDeleted]):
    event_type: Literal[InstanceEventTypes.InstanceDeleted] = (
        InstanceEventTypes.InstanceDeleted
    )
    instance_id: InstanceId


class InstanceRunnerStateUpdated(Event[InstanceEventTypes.InstanceRunnerStateUpdated]):
    event_type: Literal[InstanceEventTypes.InstanceRunnerStateUpdated] = (
        InstanceEventTypes.InstanceRunnerStateUpdated
    )
    instance_id: InstanceId
    state_update: Tuple[RunnerId, RunnerState[RunnerStateType]]


class InstanceReplacedAtomically(Event[InstanceEventTypes.InstanceReplacedAtomically]):
    event_type: Literal[InstanceEventTypes.InstanceReplacedAtomically] = (
        InstanceEventTypes.InstanceReplacedAtomically
    )
    old_instance_id: InstanceId
    new_instance_id: InstanceId
    new_instance_data: InstanceData


class MLXInferenceSagaPrepare(Event[MLXEventTypes.MLXInferenceSagaPrepare]):
    event_type: Literal[MLXEventTypes.MLXInferenceSagaPrepare] = (
        MLXEventTypes.MLXInferenceSagaPrepare
    )
    task_id: TaskId
    instance_id: InstanceId


class MLXInferenceSagaStartPrepare(Event[MLXEventTypes.MLXInferenceSagaStartPrepare]):
    event_type: Literal[MLXEventTypes.MLXInferenceSagaStartPrepare] = (
        MLXEventTypes.MLXInferenceSagaStartPrepare
    )
    task_id: TaskId
    instance_id: InstanceId


class NodeProfileUpdated(Event[NodeEventTypes.NodeProfileUpdated]):
    event_type: Literal[NodeEventTypes.NodeProfileUpdated] = (
        NodeEventTypes.NodeProfileUpdated
    )
    node_id: NodeId
    node_profile: NodeProfile


class NodeStateUpdated(Event[NodeEventTypes.NodeStateUpdated]):
    event_type: Literal[NodeEventTypes.NodeStateUpdated] = (
        NodeEventTypes.NodeStateUpdated
    )
    node_id: NodeId
    node_state: NodeState


class ChunkGenerated(Event[StreamingEventTypes.ChunkGenerated]):
    event_type: Literal[StreamingEventTypes.ChunkGenerated] = (
        StreamingEventTypes.ChunkGenerated
    )
    task_id: TaskId
    instance_id: InstanceId
    chunk: Any


class TimerScheduled(Event[TimerEventTypes.TimerCreated]):
    event_type: Literal[TimerEventTypes.TimerCreated] = TimerEventTypes.TimerCreated
    timer_data: TimerData


class TimerFired(Event[TimerEventTypes.TimerFired]):
    event_type: Literal[TimerEventTypes.TimerFired] = TimerEventTypes.TimerFired
    timer_data: TimerData
