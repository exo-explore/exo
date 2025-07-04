from __future__ import annotations

from typing import Any, Literal, Tuple

from pydantic import BaseModel

from shared.types.common import NewUUID, NodeId
from shared.types.events.common import (
    ControlPlaneEventTypes,
    DataPlaneEventTypes,
    Event,
    InstanceEventTypes,
    InstanceStateEventTypes,
    MLXEventTypes,
    NodePerformanceEventTypes,
    StreamingEventTypes,
    TaskEventTypes,
    TimerEventTypes,
)
from shared.types.networking.control_plane import (
    ControlPlaneEdgeId,
    ControlPlaneEdgeType,
)
from shared.types.networking.data_plane import (
    AddressingProtocol,
    ApplicationProtocol,
    DataPlaneEdge,
    DataPlaneEdgeId,
    DataPlaneEdgeInfoType,
    DataPlaneEdgeProfile,
)
from shared.types.profiling.common import NodePerformanceProfile
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


class RequestId(NewUUID):
    pass


class TimerId(NewUUID):
    pass


class TimerData(BaseModel):
    timer_id: TimerId


class TaskCreated[TaskTypeT: TaskType](Event[TaskEventTypes.TaskCreated]):
    event_type: Literal[TaskEventTypes.TaskCreated] = TaskEventTypes.TaskCreated
    task_id: TaskId
    task_data: TaskData[TaskTypeT]
    task_state: TaskState[TaskTypeT, Literal[TaskStatusIncompleteType.Pending]]
    on_instance: InstanceId


class TaskUpdated[TaskTypeT: TaskType](Event[TaskEventTypes.TaskUpdated]):
    event_type: Literal[TaskEventTypes.TaskUpdated] = TaskEventTypes.TaskUpdated
    task_id: TaskId
    update_data: TaskState[TaskTypeT, TaskStatusType]


class TaskDeleted(Event[TaskEventTypes.TaskDeleted]):
    event_type: Literal[TaskEventTypes.TaskDeleted] = TaskEventTypes.TaskDeleted
    task_id: TaskId


class InstanceCreated(Event[InstanceEventTypes.InstanceCreated]):
    event_type: Literal[InstanceEventTypes.InstanceCreated] = (
        InstanceEventTypes.InstanceCreated
    )
    instance_id: InstanceId
    instance_data: InstanceData
    target_status: InstanceStatus


class InstanceDeleted(Event[InstanceEventTypes.InstanceDeleted]):
    event_type: Literal[InstanceEventTypes.InstanceDeleted] = (
        InstanceEventTypes.InstanceDeleted
    )
    instance_id: InstanceId


class InstanceStatusUpdated(Event[InstanceEventTypes.InstanceStatusUpdated]):
    event_type: Literal[InstanceEventTypes.InstanceStatusUpdated] = (
        InstanceEventTypes.InstanceStatusUpdated
    )
    instance_id: InstanceId
    instance_status: InstanceStatus


class InstanceRunnerStateUpdated(
    Event[InstanceStateEventTypes.InstanceRunnerStateUpdated]
):
    event_type: Literal[InstanceStateEventTypes.InstanceRunnerStateUpdated] = (
        InstanceStateEventTypes.InstanceRunnerStateUpdated
    )
    instance_id: InstanceId
    state_update: Tuple[RunnerId, RunnerState[RunnerStateType]]


class InstanceToBeReplacedAtomically(
    Event[InstanceEventTypes.InstanceToBeReplacedAtomically]
):
    transition: Tuple[InstanceId, InstanceId]


class InstanceReplacedAtomically(Event[InstanceEventTypes.InstanceReplacedAtomically]):
    event_type: Literal[InstanceEventTypes.InstanceReplacedAtomically] = (
        InstanceEventTypes.InstanceReplacedAtomically
    )
    transition: Tuple[InstanceId, InstanceId]


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


class NodePerformanceProfiled(Event[NodePerformanceEventTypes.NodePerformanceProfiled]):
    event_type: Literal[NodePerformanceEventTypes.NodePerformanceProfiled] = (
        NodePerformanceEventTypes.NodePerformanceProfiled
    )
    node_id: NodeId
    node_profile: NodePerformanceProfile


class WorkerConnected(Event[ControlPlaneEventTypes.WorkerConnected]):
    event_type: Literal[ControlPlaneEventTypes.WorkerConnected] = (
        ControlPlaneEventTypes.WorkerConnected
    )
    edge: DataPlaneEdge[AddressingProtocol, ApplicationProtocol]


class WorkerStatusUpdated(Event[ControlPlaneEventTypes.WorkerStatusUpdated]):
    event_type: Literal[ControlPlaneEventTypes.WorkerStatusUpdated] = (
        ControlPlaneEventTypes.WorkerStatusUpdated
    )
    node_id: NodeId
    node_state: NodeStatus


class WorkerDisconnected(Event[ControlPlaneEventTypes.WorkerConnected]):
    event_type: Literal[ControlPlaneEventTypes.WorkerConnected] = (
        ControlPlaneEventTypes.WorkerConnected
    )
    vertex_id: ControlPlaneEdgeId


class ChunkGenerated(Event[StreamingEventTypes.ChunkGenerated]):
    event_type: Literal[StreamingEventTypes.ChunkGenerated] = (
        StreamingEventTypes.ChunkGenerated
    )
    task_id: TaskId
    instance_id: InstanceId
    chunk: Any


class DataPlaneEdgeCreated(Event[DataPlaneEventTypes.DataPlaneEdgeCreated]):
    event_type: Literal[DataPlaneEventTypes.DataPlaneEdgeCreated] = (
        DataPlaneEventTypes.DataPlaneEdgeCreated
    )
    vertex: ControlPlaneEdgeType


class DataPlaneEdgeProfiled(Event[DataPlaneEventTypes.DataPlaneEdgeProfiled]):
    event_type: Literal[DataPlaneEventTypes.DataPlaneEdgeProfiled] = (
        DataPlaneEventTypes.DataPlaneEdgeProfiled
    )
    edge_profile: DataPlaneEdgeProfile[Literal[DataPlaneEdgeInfoType.network_profile]]


class DataPlaneEdgeDeleted(Event[DataPlaneEventTypes.DataPlaneEdgeDeleted]):
    event_type: Literal[DataPlaneEventTypes.DataPlaneEdgeDeleted] = (
        DataPlaneEventTypes.DataPlaneEdgeDeleted
    )
    edge_id: DataPlaneEdgeId


class TimerScheduled(Event[TimerEventTypes.TimerCreated]):
    event_type: Literal[TimerEventTypes.TimerCreated] = TimerEventTypes.TimerCreated
    timer_data: TimerData


class TimerFired(Event[TimerEventTypes.TimerFired]):
    event_type: Literal[TimerEventTypes.TimerFired] = TimerEventTypes.TimerFired
    timer_data: TimerData
