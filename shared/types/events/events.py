from __future__ import annotations

from typing import Literal, Tuple

from shared.types.common import NodeId
from shared.types.events.chunks import GenerationChunk
from shared.types.events.common import (
    ControlPlaneEventTypes,
    DataPlaneEventTypes,
    Event,
    EventCategoryEnum,
    EventTypes,
    InstanceEventTypes,
    NodePerformanceEventTypes,
    RunnerStatusEventTypes,
    StreamingEventTypes,
    TaskEventTypes,
    TaskSagaEventTypes,
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
from shared.types.profiling.common import NodePerformanceProfile
from shared.types.tasks.common import (
    TaskId,
    TaskParams,
    TaskState,
    TaskStatusOtherType,
    TaskStatusType,
    TaskType,
)
from shared.types.worker.common import InstanceId, NodeStatus
from shared.types.worker.instances import InstanceParams, TypeOfInstance
from shared.types.worker.runners import RunnerId, RunnerStatus, RunnerStatusType

MLXEvent = Event[
    frozenset(
        (
            EventCategoryEnum.MutatesTaskState,
            EventCategoryEnum.MutatesControlPlaneState,
        )
    )
]
TaskEvent = Event[EventCategoryEnum.MutatesTaskState]
InstanceEvent = Event[EventCategoryEnum.MutatesInstanceState]
ControlPlaneEvent = Event[EventCategoryEnum.MutatesControlPlaneState]
DataPlaneEvent = Event[EventCategoryEnum.MutatesDataPlaneState]
NodePerformanceEvent = Event[EventCategoryEnum.MutatesNodePerformanceState]


class TaskCreated(Event[EventCategoryEnum.MutatesTaskState]):
    event_type: EventTypes = TaskEventTypes.TaskCreated
    task_id: TaskId
    task_params: TaskParams[TaskType]
    task_state: TaskState[Literal[TaskStatusOtherType.Pending], TaskType]
    on_instance: InstanceId


# Covers Cancellation Of Task, Non-Cancelled Tasks Perist
class TaskDeleted(Event[EventCategoryEnum.MutatesTaskState]):
    event_type: EventTypes = TaskEventTypes.TaskDeleted
    task_id: TaskId


class TaskStateUpdated(Event[EventCategoryEnum.MutatesTaskState]):
    event_type: EventTypes = TaskEventTypes.TaskStateUpdated
    task_state: TaskState[TaskStatusType, TaskType]


class InstanceCreated(Event[EventCategoryEnum.MutatesInstanceState]):
    event_type: EventTypes = InstanceEventTypes.InstanceCreated
    instance_id: InstanceId
    instance_params: InstanceParams
    instance_type: TypeOfInstance


class InstanceActivated(Event[EventCategoryEnum.MutatesInstanceState]):
    event_type: EventTypes = InstanceEventTypes.InstanceActivated
    instance_id: InstanceId


class InstanceDeactivated(Event[EventCategoryEnum.MutatesInstanceState]):
    event_type: EventTypes = InstanceEventTypes.InstanceDeactivated
    instance_id: InstanceId


class InstanceDeleted(Event[EventCategoryEnum.MutatesInstanceState]):
    event_type: EventTypes = InstanceEventTypes.InstanceDeleted
    instance_id: InstanceId

    transition: Tuple[InstanceId, InstanceId]


class InstanceReplacedAtomically(Event[EventCategoryEnum.MutatesInstanceState]):
    event_type: EventTypes = InstanceEventTypes.InstanceReplacedAtomically
    instance_to_replace: InstanceId
    new_instance_id: InstanceId


class RunnerStatusUpdated(Event[EventCategoryEnum.MutatesRunnerStatus]):
    event_type: EventTypes = RunnerStatusEventTypes.RunnerStatusUpdated
    instance_id: InstanceId
    state_update: Tuple[RunnerId, RunnerStatus[RunnerStatusType]]


class MLXInferenceSagaPrepare(Event[EventCategoryEnum.MutatesTaskSagaState]):
    event_type: EventTypes = TaskSagaEventTypes.MLXInferenceSagaPrepare
    task_id: TaskId
    instance_id: InstanceId


class MLXInferenceSagaStartPrepare(Event[EventCategoryEnum.MutatesTaskSagaState]):
    event_type: EventTypes = TaskSagaEventTypes.MLXInferenceSagaStartPrepare
    task_id: TaskId
    instance_id: InstanceId


class NodePerformanceMeasured(Event[EventCategoryEnum.MutatesNodePerformanceState]):
    event_type: EventTypes = NodePerformanceEventTypes.NodePerformanceMeasured
    node_id: NodeId
    node_profile: NodePerformanceProfile


class WorkerConnected(Event[EventCategoryEnum.MutatesControlPlaneState]):
    event_type: EventTypes = ControlPlaneEventTypes.WorkerConnected
    edge: DataPlaneEdge


class WorkerStatusUpdated(Event[EventCategoryEnum.MutatesControlPlaneState]):
    event_type: EventTypes = ControlPlaneEventTypes.WorkerStatusUpdated
    node_id: NodeId
    node_state: NodeStatus


class WorkerDisconnected(Event[EventCategoryEnum.MutatesControlPlaneState]):
    event_type: EventTypes = ControlPlaneEventTypes.WorkerConnected
    vertex_id: ControlPlaneEdgeId


class ChunkGenerated(Event[EventCategoryEnum.MutatesTaskState]):
    event_type: EventTypes = StreamingEventTypes.ChunkGenerated
    task_id: TaskId
    chunk: GenerationChunk


class DataPlaneEdgeCreated(Event[EventCategoryEnum.MutatesDataPlaneState]):
    event_type: EventTypes = DataPlaneEventTypes.DataPlaneEdgeCreated
    vertex: ControlPlaneEdgeType


class DataPlaneEdgeReplacedAtomically(Event[EventCategoryEnum.MutatesDataPlaneState]):
    event_type: EventTypes = DataPlaneEventTypes.DataPlaneEdgeReplacedAtomically
    edge_id: DataPlaneEdgeId
    edge_profile: DataPlaneEdgeProfile


class DataPlaneEdgeDeleted(Event[EventCategoryEnum.MutatesDataPlaneState]):
    event_type: EventTypes = DataPlaneEventTypes.DataPlaneEdgeDeleted
    edge_id: DataPlaneEdgeId
