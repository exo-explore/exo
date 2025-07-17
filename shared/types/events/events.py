from __future__ import annotations

from typing import Literal, Tuple

from shared.types.common import NodeId
from shared.types.events.chunks import GenerationChunk
from shared.types.events.common import (
    BaseEvent,
    ControlPlaneEventTypes,
    DataPlaneEventTypes,
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

TaskEvent = BaseEvent[EventCategoryEnum.MutatesTaskState]
InstanceEvent = BaseEvent[EventCategoryEnum.MutatesInstanceState]
ControlPlaneEvent = BaseEvent[EventCategoryEnum.MutatesControlPlaneState]
DataPlaneEvent = BaseEvent[EventCategoryEnum.MutatesDataPlaneState]
NodePerformanceEvent = BaseEvent[EventCategoryEnum.MutatesNodePerformanceState]


class TaskCreated(BaseEvent[EventCategoryEnum.MutatesTaskState]):
    event_type: EventTypes = TaskEventTypes.TaskCreated
    event_category: Literal[EventCategoryEnum.MutatesTaskState] = EventCategoryEnum.MutatesTaskState
    task_id: TaskId
    task_params: TaskParams[TaskType]
    task_state: TaskState[Literal[TaskStatusOtherType.Pending], TaskType]
    on_instance: InstanceId


# Covers Cancellation Of Task, Non-Cancelled Tasks Perist
class TaskDeleted(BaseEvent[EventCategoryEnum.MutatesTaskState]):
    event_type: EventTypes = TaskEventTypes.TaskDeleted
    event_category: Literal[EventCategoryEnum.MutatesTaskState] = EventCategoryEnum.MutatesTaskState
    task_id: TaskId


class TaskStateUpdated(BaseEvent[EventCategoryEnum.MutatesTaskState]):
    event_type: EventTypes = TaskEventTypes.TaskStateUpdated
    event_category: Literal[EventCategoryEnum.MutatesTaskState] = EventCategoryEnum.MutatesTaskState
    task_state: TaskState[TaskStatusType, TaskType]


class InstanceCreated(BaseEvent[EventCategoryEnum.MutatesInstanceState]):
    event_type: EventTypes = InstanceEventTypes.InstanceCreated
    event_category: Literal[EventCategoryEnum.MutatesInstanceState] = EventCategoryEnum.MutatesInstanceState
    instance_id: InstanceId
    instance_params: InstanceParams
    instance_type: TypeOfInstance


class InstanceActivated(BaseEvent[EventCategoryEnum.MutatesInstanceState]):
    event_type: EventTypes = InstanceEventTypes.InstanceActivated
    event_category: Literal[EventCategoryEnum.MutatesInstanceState] = EventCategoryEnum.MutatesInstanceState
    instance_id: InstanceId


class InstanceDeactivated(BaseEvent[EventCategoryEnum.MutatesInstanceState]):
    event_type: EventTypes = InstanceEventTypes.InstanceDeactivated
    event_category: Literal[EventCategoryEnum.MutatesInstanceState] = EventCategoryEnum.MutatesInstanceState
    instance_id: InstanceId


class InstanceDeleted(BaseEvent[EventCategoryEnum.MutatesInstanceState]):
    event_type: EventTypes = InstanceEventTypes.InstanceDeleted
    event_category: Literal[EventCategoryEnum.MutatesInstanceState] = EventCategoryEnum.MutatesInstanceState
    instance_id: InstanceId

    transition: Tuple[InstanceId, InstanceId]


class InstanceReplacedAtomically(BaseEvent[EventCategoryEnum.MutatesInstanceState]):
    event_type: EventTypes = InstanceEventTypes.InstanceReplacedAtomically
    event_category: Literal[EventCategoryEnum.MutatesInstanceState] = EventCategoryEnum.MutatesInstanceState
    instance_to_replace: InstanceId
    new_instance_id: InstanceId


class RunnerStatusUpdated(BaseEvent[EventCategoryEnum.MutatesRunnerStatus]):
    event_type: EventTypes = RunnerStatusEventTypes.RunnerStatusUpdated
    event_category: Literal[EventCategoryEnum.MutatesRunnerStatus] = EventCategoryEnum.MutatesRunnerStatus
    instance_id: InstanceId
    state_update: Tuple[RunnerId, RunnerStatus[RunnerStatusType]]


class MLXInferenceSagaPrepare(BaseEvent[EventCategoryEnum.MutatesTaskSagaState]):
    event_type: EventTypes = TaskSagaEventTypes.MLXInferenceSagaPrepare
    event_category: Literal[EventCategoryEnum.MutatesTaskSagaState] = EventCategoryEnum.MutatesTaskSagaState
    task_id: TaskId
    instance_id: InstanceId


class MLXInferenceSagaStartPrepare(BaseEvent[EventCategoryEnum.MutatesTaskSagaState]):
    event_type: EventTypes = TaskSagaEventTypes.MLXInferenceSagaStartPrepare
    event_category: Literal[EventCategoryEnum.MutatesTaskSagaState] = EventCategoryEnum.MutatesTaskSagaState
    task_id: TaskId
    instance_id: InstanceId


class NodePerformanceMeasured(BaseEvent[EventCategoryEnum.MutatesNodePerformanceState]):
    event_type: EventTypes = NodePerformanceEventTypes.NodePerformanceMeasured
    event_category: Literal[EventCategoryEnum.MutatesNodePerformanceState] = EventCategoryEnum.MutatesNodePerformanceState
    node_id: NodeId
    node_profile: NodePerformanceProfile


class WorkerConnected(BaseEvent[EventCategoryEnum.MutatesControlPlaneState]):
    event_type: EventTypes = ControlPlaneEventTypes.WorkerConnected
    event_category: Literal[EventCategoryEnum.MutatesControlPlaneState] = EventCategoryEnum.MutatesControlPlaneState
    edge: DataPlaneEdge


class WorkerStatusUpdated(BaseEvent[EventCategoryEnum.MutatesControlPlaneState]):
    event_type: EventTypes = ControlPlaneEventTypes.WorkerStatusUpdated
    event_category: Literal[EventCategoryEnum.MutatesControlPlaneState] = EventCategoryEnum.MutatesControlPlaneState
    node_id: NodeId
    node_state: NodeStatus


class WorkerDisconnected(BaseEvent[EventCategoryEnum.MutatesControlPlaneState]):
    event_type: EventTypes = ControlPlaneEventTypes.WorkerConnected
    event_category: Literal[EventCategoryEnum.MutatesControlPlaneState] = EventCategoryEnum.MutatesControlPlaneState
    vertex_id: ControlPlaneEdgeId


class ChunkGenerated(BaseEvent[EventCategoryEnum.MutatesTaskState]):
    event_type: EventTypes = StreamingEventTypes.ChunkGenerated
    event_category: Literal[EventCategoryEnum.MutatesTaskState] = EventCategoryEnum.MutatesTaskState
    task_id: TaskId
    chunk: GenerationChunk


class DataPlaneEdgeCreated(BaseEvent[EventCategoryEnum.MutatesDataPlaneState]):
    event_type: EventTypes = DataPlaneEventTypes.DataPlaneEdgeCreated
    event_category: Literal[EventCategoryEnum.MutatesDataPlaneState] = EventCategoryEnum.MutatesDataPlaneState
    vertex: ControlPlaneEdgeType


class DataPlaneEdgeReplacedAtomically(BaseEvent[EventCategoryEnum.MutatesDataPlaneState]):
    event_type: EventTypes = DataPlaneEventTypes.DataPlaneEdgeReplacedAtomically
    event_category: Literal[EventCategoryEnum.MutatesDataPlaneState] = EventCategoryEnum.MutatesDataPlaneState
    edge_id: DataPlaneEdgeId
    edge_profile: DataPlaneEdgeProfile


class DataPlaneEdgeDeleted(BaseEvent[EventCategoryEnum.MutatesDataPlaneState]):
    event_type: EventTypes = DataPlaneEventTypes.DataPlaneEdgeDeleted
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