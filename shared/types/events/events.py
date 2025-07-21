from __future__ import annotations

from typing import Literal, Tuple

from shared.types.common import NodeId
from shared.types.events.chunks import GenerationChunk
from shared.types.events.common import (
    BaseEvent,
    EventCategoryEnum,
    InstanceEventTypes,
    NodePerformanceEventTypes,
    RunnerStatusEventTypes,
    StreamingEventTypes,
    TaskEventTypes,
    TaskSagaEventTypes,
    TopologyEventTypes,
)
from shared.types.graphs.topology import (
    TopologyEdge,
    TopologyEdgeId,
    TopologyEdgeProfile,
    TopologyNode,
)
from shared.types.profiling.common import NodePerformanceProfile
from shared.types.tasks.common import Task, TaskId, TaskStatus
from shared.types.worker.common import InstanceId, NodeStatus
from shared.types.worker.instances import InstanceParams, TypeOfInstance
from shared.types.worker.runners import RunnerId, RunnerStatus

TaskEvent = BaseEvent[EventCategoryEnum.MutatesTaskState]
InstanceEvent = BaseEvent[EventCategoryEnum.MutatesInstanceState]
TopologyEvent = BaseEvent[EventCategoryEnum.MutatesTopologyState]
NodePerformanceEvent = BaseEvent[EventCategoryEnum.MutatesNodePerformanceState]


class TaskCreated(BaseEvent[EventCategoryEnum.MutatesTaskState, Literal[TaskEventTypes.TaskCreated]]):
    event_type: Literal[TaskEventTypes.TaskCreated] = TaskEventTypes.TaskCreated
    event_category: Literal[EventCategoryEnum.MutatesTaskState] = EventCategoryEnum.MutatesTaskState
    task_id: TaskId
    task: Task


# Covers Cancellation Of Task, Non-Cancelled Tasks Perist
class TaskDeleted(BaseEvent[EventCategoryEnum.MutatesTaskState, Literal[TaskEventTypes.TaskDeleted]]):
    event_type: Literal[TaskEventTypes.TaskDeleted] = TaskEventTypes.TaskDeleted
    event_category: Literal[EventCategoryEnum.MutatesTaskState] = EventCategoryEnum.MutatesTaskState
    task_id: TaskId


class TaskStateUpdated(BaseEvent[EventCategoryEnum.MutatesTaskState, Literal[TaskEventTypes.TaskStateUpdated]]):
    event_type: Literal[TaskEventTypes.TaskStateUpdated] = TaskEventTypes.TaskStateUpdated
    event_category: Literal[EventCategoryEnum.MutatesTaskState] = EventCategoryEnum.MutatesTaskState
    task_id: TaskId
    task_status: TaskStatus


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


class WorkerConnected(BaseEvent[EventCategoryEnum.MutatesTopologyState, Literal[TopologyEventTypes.WorkerConnected]]):
    event_type: Literal[TopologyEventTypes.WorkerConnected] = TopologyEventTypes.WorkerConnected
    event_category: Literal[EventCategoryEnum.MutatesTopologyState] = EventCategoryEnum.MutatesTopologyState
    edge: TopologyEdge


class WorkerStatusUpdated(BaseEvent[EventCategoryEnum.MutatesTopologyState, Literal[TopologyEventTypes.WorkerStatusUpdated]]):
    event_type: Literal[TopologyEventTypes.WorkerStatusUpdated] = TopologyEventTypes.WorkerStatusUpdated
    event_category: Literal[EventCategoryEnum.MutatesTopologyState] = EventCategoryEnum.MutatesTopologyState
    node_id: NodeId
    node_state: NodeStatus


class WorkerDisconnected(BaseEvent[EventCategoryEnum.MutatesTopologyState, Literal[TopologyEventTypes.WorkerDisconnected]]):
    event_type: Literal[TopologyEventTypes.WorkerDisconnected] = TopologyEventTypes.WorkerDisconnected
    event_category: Literal[EventCategoryEnum.MutatesTopologyState] = EventCategoryEnum.MutatesTopologyState
    vertex_id: NodeId


class ChunkGenerated(BaseEvent[EventCategoryEnum.MutatesTaskState, Literal[StreamingEventTypes.ChunkGenerated]]):
    event_type: Literal[StreamingEventTypes.ChunkGenerated] = StreamingEventTypes.ChunkGenerated
    event_category: Literal[EventCategoryEnum.MutatesTaskState] = EventCategoryEnum.MutatesTaskState
    task_id: TaskId
    chunk: GenerationChunk


class TopologyEdgeCreated(BaseEvent[EventCategoryEnum.MutatesTopologyState, Literal[TopologyEventTypes.TopologyEdgeCreated]]):
    event_type: Literal[TopologyEventTypes.TopologyEdgeCreated] = TopologyEventTypes.TopologyEdgeCreated
    event_category: Literal[EventCategoryEnum.MutatesTopologyState] = EventCategoryEnum.MutatesTopologyState
    vertex: TopologyNode


class TopologyEdgeReplacedAtomically(BaseEvent[EventCategoryEnum.MutatesTopologyState, Literal[TopologyEventTypes.TopologyEdgeReplacedAtomically]]):
    event_type: Literal[TopologyEventTypes.TopologyEdgeReplacedAtomically] = TopologyEventTypes.TopologyEdgeReplacedAtomically
    event_category: Literal[EventCategoryEnum.MutatesTopologyState] = EventCategoryEnum.MutatesTopologyState
    edge_id: TopologyEdgeId
    edge_profile: TopologyEdgeProfile


class TopologyEdgeDeleted(BaseEvent[EventCategoryEnum.MutatesTopologyState, Literal[TopologyEventTypes.TopologyEdgeDeleted]]):
    event_type: Literal[TopologyEventTypes.TopologyEdgeDeleted] = TopologyEventTypes.TopologyEdgeDeleted
    event_category: Literal[EventCategoryEnum.MutatesTopologyState] = EventCategoryEnum.MutatesTopologyState
    edge_id: TopologyEdgeId

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