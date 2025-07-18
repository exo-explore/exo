from __future__ import annotations

from typing import Literal, Tuple

from shared.types.common import NodeId
from shared.types.events.chunks import GenerationChunk
from shared.types.events.common import (
    BaseEvent,
    ControlPlaneEventTypes,
    DataPlaneEventTypes,
    EventCategoryEnum,
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
    BaseTaskData,
    TaskId,
    TaskState,
    TaskStatusOtherType,
    TaskStatusType,
    TaskType,
)
from shared.types.worker.common import InstanceId, NodeStatus
from shared.types.worker.instances import InstanceParams, TypeOfInstance
from shared.types.worker.runners import RunnerId, RunnerStatus

TaskEvent = BaseEvent[EventCategoryEnum.MutatesTaskState]
InstanceEvent = BaseEvent[EventCategoryEnum.MutatesInstanceState]
ControlPlaneEvent = BaseEvent[EventCategoryEnum.MutatesControlPlaneState]
DataPlaneEvent = BaseEvent[EventCategoryEnum.MutatesDataPlaneState]
NodePerformanceEvent = BaseEvent[EventCategoryEnum.MutatesNodePerformanceState]


class TaskCreated(BaseEvent[EventCategoryEnum.MutatesTaskState, Literal[TaskEventTypes.TaskCreated]]):
    event_type: Literal[TaskEventTypes.TaskCreated] = TaskEventTypes.TaskCreated
    event_category: Literal[EventCategoryEnum.MutatesTaskState] = EventCategoryEnum.MutatesTaskState
    task_id: TaskId
    task_data: BaseTaskData[TaskType]
    task_state: TaskState[Literal[TaskStatusOtherType.Pending], TaskType]
    on_instance: InstanceId


# Covers Cancellation Of Task, Non-Cancelled Tasks Perist
class TaskDeleted(BaseEvent[EventCategoryEnum.MutatesTaskState, Literal[TaskEventTypes.TaskDeleted]]):
    event_type: Literal[TaskEventTypes.TaskDeleted] = TaskEventTypes.TaskDeleted
    event_category: Literal[EventCategoryEnum.MutatesTaskState] = EventCategoryEnum.MutatesTaskState
    task_id: TaskId


class TaskStateUpdated(BaseEvent[EventCategoryEnum.MutatesTaskState, Literal[TaskEventTypes.TaskStateUpdated]]):
    event_type: Literal[TaskEventTypes.TaskStateUpdated] = TaskEventTypes.TaskStateUpdated
    event_category: Literal[EventCategoryEnum.MutatesTaskState] = EventCategoryEnum.MutatesTaskState
    task_state: TaskState[TaskStatusType, TaskType]


class InstanceCreated(BaseEvent[EventCategoryEnum.MutatesInstanceState, Literal[InstanceEventTypes.InstanceCreated]]):
    event_type: Literal[InstanceEventTypes.InstanceCreated] = InstanceEventTypes.InstanceCreated
    event_category: Literal[EventCategoryEnum.MutatesInstanceState] = EventCategoryEnum.MutatesInstanceState
    instance_id: InstanceId
    instance_params: InstanceParams
    instance_type: TypeOfInstance


class InstanceActivated(BaseEvent[EventCategoryEnum.MutatesInstanceState, Literal[InstanceEventTypes.InstanceActivated]]):
    event_type: Literal[InstanceEventTypes.InstanceActivated] = InstanceEventTypes.InstanceActivated
    event_category: Literal[EventCategoryEnum.MutatesInstanceState] = EventCategoryEnum.MutatesInstanceState
    instance_id: InstanceId


class InstanceDeactivated(BaseEvent[EventCategoryEnum.MutatesInstanceState, Literal[InstanceEventTypes.InstanceDeactivated]]):
    event_type: Literal[InstanceEventTypes.InstanceDeactivated] = InstanceEventTypes.InstanceDeactivated
    event_category: Literal[EventCategoryEnum.MutatesInstanceState] = EventCategoryEnum.MutatesInstanceState
    instance_id: InstanceId


class InstanceDeleted(BaseEvent[EventCategoryEnum.MutatesInstanceState, Literal[InstanceEventTypes.InstanceDeleted]]):
    event_type: Literal[InstanceEventTypes.InstanceDeleted] = InstanceEventTypes.InstanceDeleted
    event_category: Literal[EventCategoryEnum.MutatesInstanceState] = EventCategoryEnum.MutatesInstanceState
    instance_id: InstanceId

    transition: Tuple[InstanceId, InstanceId]


class InstanceReplacedAtomically(BaseEvent[EventCategoryEnum.MutatesInstanceState, Literal[InstanceEventTypes.InstanceReplacedAtomically]]):
    event_type: Literal[InstanceEventTypes.InstanceReplacedAtomically] = InstanceEventTypes.InstanceReplacedAtomically
    event_category: Literal[EventCategoryEnum.MutatesInstanceState] = EventCategoryEnum.MutatesInstanceState
    instance_to_replace: InstanceId
    new_instance_id: InstanceId


class RunnerStatusUpdated(BaseEvent[EventCategoryEnum.MutatesRunnerStatus, Literal[RunnerStatusEventTypes.RunnerStatusUpdated]]):
    event_type: Literal[RunnerStatusEventTypes.RunnerStatusUpdated] = RunnerStatusEventTypes.RunnerStatusUpdated
    event_category: Literal[EventCategoryEnum.MutatesRunnerStatus] = EventCategoryEnum.MutatesRunnerStatus
    runner_id: RunnerId
    runner_status: RunnerStatus


class MLXInferenceSagaPrepare(BaseEvent[EventCategoryEnum.MutatesTaskSagaState, Literal[TaskSagaEventTypes.MLXInferenceSagaPrepare]]):
    event_type: Literal[TaskSagaEventTypes.MLXInferenceSagaPrepare] = TaskSagaEventTypes.MLXInferenceSagaPrepare
    event_category: Literal[EventCategoryEnum.MutatesTaskSagaState] = EventCategoryEnum.MutatesTaskSagaState
    task_id: TaskId
    instance_id: InstanceId


class MLXInferenceSagaStartPrepare(BaseEvent[EventCategoryEnum.MutatesTaskSagaState, Literal[TaskSagaEventTypes.MLXInferenceSagaStartPrepare]]):
    event_type: Literal[TaskSagaEventTypes.MLXInferenceSagaStartPrepare] = TaskSagaEventTypes.MLXInferenceSagaStartPrepare
    event_category: Literal[EventCategoryEnum.MutatesTaskSagaState] = EventCategoryEnum.MutatesTaskSagaState
    task_id: TaskId
    instance_id: InstanceId


class NodePerformanceMeasured(BaseEvent[EventCategoryEnum.MutatesNodePerformanceState, Literal[NodePerformanceEventTypes.NodePerformanceMeasured]]):
    event_type: Literal[NodePerformanceEventTypes.NodePerformanceMeasured] = NodePerformanceEventTypes.NodePerformanceMeasured
    event_category: Literal[EventCategoryEnum.MutatesNodePerformanceState] = EventCategoryEnum.MutatesNodePerformanceState
    node_id: NodeId
    node_profile: NodePerformanceProfile


class WorkerConnected(BaseEvent[EventCategoryEnum.MutatesControlPlaneState, Literal[ControlPlaneEventTypes.WorkerConnected]]):
    event_type: Literal[ControlPlaneEventTypes.WorkerConnected] = ControlPlaneEventTypes.WorkerConnected
    event_category: Literal[EventCategoryEnum.MutatesControlPlaneState] = EventCategoryEnum.MutatesControlPlaneState
    edge: DataPlaneEdge


class WorkerStatusUpdated(BaseEvent[EventCategoryEnum.MutatesControlPlaneState, Literal[ControlPlaneEventTypes.WorkerStatusUpdated]]):
    event_type: Literal[ControlPlaneEventTypes.WorkerStatusUpdated] = ControlPlaneEventTypes.WorkerStatusUpdated
    event_category: Literal[EventCategoryEnum.MutatesControlPlaneState] = EventCategoryEnum.MutatesControlPlaneState
    node_id: NodeId
    node_state: NodeStatus


class WorkerDisconnected(BaseEvent[EventCategoryEnum.MutatesControlPlaneState, Literal[ControlPlaneEventTypes.WorkerDisconnected]]):
    event_type: Literal[ControlPlaneEventTypes.WorkerDisconnected] = ControlPlaneEventTypes.WorkerDisconnected
    event_category: Literal[EventCategoryEnum.MutatesControlPlaneState] = EventCategoryEnum.MutatesControlPlaneState
    vertex_id: ControlPlaneEdgeId


class ChunkGenerated(BaseEvent[EventCategoryEnum.MutatesTaskState, Literal[StreamingEventTypes.ChunkGenerated]]):
    event_type: Literal[StreamingEventTypes.ChunkGenerated] = StreamingEventTypes.ChunkGenerated
    event_category: Literal[EventCategoryEnum.MutatesTaskState] = EventCategoryEnum.MutatesTaskState
    task_id: TaskId
    chunk: GenerationChunk


class DataPlaneEdgeCreated(BaseEvent[EventCategoryEnum.MutatesDataPlaneState, Literal[DataPlaneEventTypes.DataPlaneEdgeCreated]]):
    event_type: Literal[DataPlaneEventTypes.DataPlaneEdgeCreated] = DataPlaneEventTypes.DataPlaneEdgeCreated
    event_category: Literal[EventCategoryEnum.MutatesDataPlaneState] = EventCategoryEnum.MutatesDataPlaneState
    vertex: ControlPlaneEdgeType


class DataPlaneEdgeReplacedAtomically(BaseEvent[EventCategoryEnum.MutatesDataPlaneState, Literal[DataPlaneEventTypes.DataPlaneEdgeReplacedAtomically]]):
    event_type: Literal[DataPlaneEventTypes.DataPlaneEdgeReplacedAtomically] = DataPlaneEventTypes.DataPlaneEdgeReplacedAtomically
    event_category: Literal[EventCategoryEnum.MutatesDataPlaneState] = EventCategoryEnum.MutatesDataPlaneState
    edge_id: DataPlaneEdgeId
    edge_profile: DataPlaneEdgeProfile


class DataPlaneEdgeDeleted(BaseEvent[EventCategoryEnum.MutatesDataPlaneState, Literal[DataPlaneEventTypes.DataPlaneEdgeDeleted]]):
    event_type: Literal[DataPlaneEventTypes.DataPlaneEdgeDeleted] = DataPlaneEventTypes.DataPlaneEdgeDeleted
    event_category: Literal[EventCategoryEnum.MutatesDataPlaneState] = EventCategoryEnum.MutatesDataPlaneState
    edge_id: DataPlaneEdgeId

"""
TEST_EVENT_CATEGORIES_TYPE = FrozenSet[
    Literal[
        EventCategoryEnum.MutatesTaskState,
        EventCategoryEnum.MutatesControlPlaneState,
    ]
]
TEST_EVENT_CATEGORIES = frozenset(
    (
        EventCategoryEnum.MutatesTaskState,
        EventCategoryEnum.MutatesControlPlaneState,
    )
)


class TestEvent(BaseEvent[TEST_EVENT_CATEGORIES_TYPE]):
    event_category: TEST_EVENT_CATEGORIES_TYPE = TEST_EVENT_CATEGORIES
    test_id: int
"""