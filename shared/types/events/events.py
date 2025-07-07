from __future__ import annotations

from typing import Any, Literal, Tuple

from pydantic import BaseModel

from shared.types.common import NodeId
from shared.types.events.common import (
    ControlPlaneEvent,
    ControlPlaneEventTypes,
    DataPlaneEvent,
    DataPlaneEventTypes,
    InstanceEvent,
    InstanceEventTypes,
    InstanceStateEvent,
    InstanceStateEventTypes,
    MLXEvent,
    MLXEventTypes,
    NodePerformanceEvent,
    NodePerformanceEventTypes,
    ResourceEvent,
    ResourceEventTypes,
    StreamingEvent,
    StreamingEventTypes,
    TaskEvent,
    TaskEventTypes,
    TimerEvent,
    TimerEventTypes,
    TimerId,
)
from shared.types.networking.control_plane import (
    ControlPlaneEdgeId,
    ControlPlaneEdgeType,
)
from shared.types.networking.data_plane import (
    DataPlaneEdge,
    DataPlaneEdgeId,
    DataPlaneEdgeProfile,
)
from shared.types.profiling.common import NodePerformanceProfile, ProfiledResourceName
from shared.types.tasks.common import (
    TaskData,
    TaskId,
    TaskState,
    TaskStatusIncompleteType,
    TaskStatusType,
    TaskType,
)
from shared.types.worker.common import InstanceId, NodeStatus
from shared.types.worker.instances import InstanceData, InstanceStatus
from shared.types.worker.runners import RunnerId, RunnerState, RunnerStateType


class TimerData(BaseModel):
    timer_id: TimerId


class TaskCreated[TaskTypeT: TaskType](TaskEvent):
    event_type: TaskEventTypes = TaskEventTypes.TaskCreated
    task_id: TaskId
    task_data: TaskData[TaskTypeT]
    task_state: TaskState[Literal[TaskStatusIncompleteType.Pending], TaskTypeT]
    on_instance: InstanceId


class TaskUpdated[TaskTypeT: TaskType](TaskEvent):
    event_type: TaskEventTypes = TaskEventTypes.TaskUpdated
    task_id: TaskId
    update_data: TaskState[TaskStatusType, TaskTypeT]


class TaskDeleted(TaskEvent):
    event_type: TaskEventTypes = TaskEventTypes.TaskDeleted
    task_id: TaskId


class InstanceCreated(InstanceEvent):
    event_type: InstanceEventTypes = InstanceEventTypes.InstanceCreated
    instance_id: InstanceId
    instance_data: InstanceData
    target_status: InstanceStatus


class InstanceDeleted(InstanceEvent):
    event_type: InstanceEventTypes = InstanceEventTypes.InstanceDeleted
    instance_id: InstanceId


class InstanceStatusUpdated(InstanceEvent):
    event_type: InstanceEventTypes = InstanceEventTypes.InstanceStatusUpdated
    instance_id: InstanceId
    instance_status: InstanceStatus


class InstanceRunnerStateUpdated(InstanceStateEvent):
    event_type: InstanceStateEventTypes = (
        InstanceStateEventTypes.InstanceRunnerStateUpdated
    )
    instance_id: InstanceId
    state_update: Tuple[RunnerId, RunnerState[RunnerStateType]]


class InstanceToBeReplacedAtomically(InstanceEvent):
    event_type: InstanceEventTypes = InstanceEventTypes.InstanceToBeReplacedAtomically
    transition: Tuple[InstanceId, InstanceId]


class InstanceReplacedAtomically(InstanceEvent):
    event_type: InstanceEventTypes = InstanceEventTypes.InstanceReplacedAtomically
    transition: Tuple[InstanceId, InstanceId]


class MLXInferenceSagaPrepare(MLXEvent):
    event_type: MLXEventTypes = MLXEventTypes.MLXInferenceSagaPrepare
    task_id: TaskId
    instance_id: InstanceId


class MLXInferenceSagaStartPrepare(MLXEvent):
    event_type: MLXEventTypes = MLXEventTypes.MLXInferenceSagaStartPrepare
    task_id: TaskId
    instance_id: InstanceId


class NodePerformanceProfiled(NodePerformanceEvent):
    event_type: NodePerformanceEventTypes = (
        NodePerformanceEventTypes.NodePerformanceProfiled
    )
    node_id: NodeId
    node_profile: NodePerformanceProfile


class WorkerConnected(ControlPlaneEvent):
    event_type: ControlPlaneEventTypes = ControlPlaneEventTypes.WorkerConnected
    edge: DataPlaneEdge


class WorkerStatusUpdated(ControlPlaneEvent):
    event_type: ControlPlaneEventTypes = ControlPlaneEventTypes.WorkerStatusUpdated
    node_id: NodeId
    node_state: NodeStatus


class WorkerDisconnected(ControlPlaneEvent):
    event_type: ControlPlaneEventTypes = ControlPlaneEventTypes.WorkerConnected
    vertex_id: ControlPlaneEdgeId


class ChunkGenerated(StreamingEvent):
    event_type: StreamingEventTypes = StreamingEventTypes.ChunkGenerated
    task_id: TaskId
    instance_id: InstanceId
    chunk: Any


class DataPlaneEdgeCreated(DataPlaneEvent):
    event_type: DataPlaneEventTypes = DataPlaneEventTypes.DataPlaneEdgeCreated
    vertex: ControlPlaneEdgeType


class DataPlaneEdgeProfiled(DataPlaneEvent):
    event_type: DataPlaneEventTypes = DataPlaneEventTypes.DataPlaneEdgeProfiled
    edge_id: DataPlaneEdgeId
    edge_profile: DataPlaneEdgeProfile


class DataPlaneEdgeDeleted(DataPlaneEvent):
    event_type: DataPlaneEventTypes = DataPlaneEventTypes.DataPlaneEdgeDeleted
    edge_id: DataPlaneEdgeId


class TimerScheduled(TimerEvent):
    event_type: TimerEventTypes = TimerEventTypes.TimerCreated
    timer_data: TimerData


class TimerFired(TimerEvent):
    event_type: TimerEventTypes = TimerEventTypes.TimerFired
    timer_data: TimerData


class ResourceProfiled(ResourceEvent):
    event_type: ResourceEventTypes = ResourceEventTypes.ResourceProfiled
    resource_name: ProfiledResourceName
    resource_profile: NodePerformanceProfile
