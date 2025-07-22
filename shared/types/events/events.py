from __future__ import annotations

from typing import Literal, Tuple

from shared.types.common import NodeId
from shared.types.events.chunks import GenerationChunk
from shared.types.events.common import (
    BaseEvent,
    EventType,
    TimerId,
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


class TaskCreated(BaseEvent[EventType.TaskCreated]):
    event_type: Literal[EventType.TaskCreated] = EventType.TaskCreated
    task_id: TaskId
    task: Task


class TaskDeleted(BaseEvent[EventType.TaskDeleted]):
    event_type: Literal[EventType.TaskDeleted] = EventType.TaskDeleted
    task_id: TaskId


class TaskStateUpdated(BaseEvent[EventType.TaskStateUpdated]):
    event_type: Literal[EventType.TaskStateUpdated] = EventType.TaskStateUpdated
    task_id: TaskId
    task_status: TaskStatus


class InstanceCreated(BaseEvent[EventType.InstanceCreated]):
    event_type: Literal[EventType.InstanceCreated] = EventType.InstanceCreated
    instance_id: InstanceId
    instance_params: InstanceParams
    instance_type: TypeOfInstance


class InstanceActivated(BaseEvent[EventType.InstanceActivated]):
    event_type: Literal[EventType.InstanceActivated] = EventType.InstanceActivated
    instance_id: InstanceId


class InstanceDeactivated(BaseEvent[EventType.InstanceDeactivated]):
    event_type: Literal[EventType.InstanceDeactivated] = EventType.InstanceDeactivated
    instance_id: InstanceId


class InstanceDeleted(BaseEvent[EventType.InstanceDeleted]):
    event_type: Literal[EventType.InstanceDeleted] = EventType.InstanceDeleted
    instance_id: InstanceId

    transition: Tuple[InstanceId, InstanceId]


class InstanceReplacedAtomically(BaseEvent[EventType.InstanceReplacedAtomically]):
    event_type: Literal[EventType.InstanceReplacedAtomically] = EventType.InstanceReplacedAtomically
    instance_to_replace: InstanceId
    new_instance_id: InstanceId


class RunnerStatusUpdated(BaseEvent[EventType.RunnerStatusUpdated]):
    event_type: Literal[EventType.RunnerStatusUpdated] = EventType.RunnerStatusUpdated
    runner_id: RunnerId
    runner_status: RunnerStatus


class MLXInferenceSagaPrepare(BaseEvent[EventType.MLXInferenceSagaPrepare]):
    event_type: Literal[EventType.MLXInferenceSagaPrepare] = EventType.MLXInferenceSagaPrepare
    task_id: TaskId
    instance_id: InstanceId


class MLXInferenceSagaStartPrepare(BaseEvent[EventType.MLXInferenceSagaStartPrepare]):
    event_type: Literal[EventType.MLXInferenceSagaStartPrepare] = EventType.MLXInferenceSagaStartPrepare
    task_id: TaskId
    instance_id: InstanceId


class NodePerformanceMeasured(BaseEvent[EventType.NodePerformanceMeasured]):
    event_type: Literal[EventType.NodePerformanceMeasured] = EventType.NodePerformanceMeasured
    node_id: NodeId
    node_profile: NodePerformanceProfile


class WorkerConnected(BaseEvent[EventType.WorkerConnected]):
    event_type: Literal[EventType.WorkerConnected] = EventType.WorkerConnected
    edge: TopologyEdge


class WorkerStatusUpdated(BaseEvent[EventType.WorkerStatusUpdated]):
    event_type: Literal[EventType.WorkerStatusUpdated] = EventType.WorkerStatusUpdated
    node_id: NodeId
    node_state: NodeStatus


class WorkerDisconnected(BaseEvent[EventType.WorkerDisconnected]):
    event_type: Literal[EventType.WorkerDisconnected] = EventType.WorkerDisconnected
    vertex_id: NodeId


class ChunkGenerated(BaseEvent[EventType.ChunkGenerated]):
    event_type: Literal[EventType.ChunkGenerated] = EventType.ChunkGenerated
    task_id: TaskId
    chunk: GenerationChunk


class TopologyEdgeCreated(BaseEvent[EventType.TopologyEdgeCreated]):
    event_type: Literal[EventType.TopologyEdgeCreated] = EventType.TopologyEdgeCreated
    vertex: TopologyNode


class TopologyEdgeReplacedAtomically(BaseEvent[EventType.TopologyEdgeReplacedAtomically]):
    event_type: Literal[EventType.TopologyEdgeReplacedAtomically] = EventType.TopologyEdgeReplacedAtomically
    edge_id: TopologyEdgeId
    edge_profile: TopologyEdgeProfile


class TopologyEdgeDeleted(BaseEvent[EventType.TopologyEdgeDeleted]):
    event_type: Literal[EventType.TopologyEdgeDeleted] = EventType.TopologyEdgeDeleted
    edge_id: TopologyEdgeId


class TimerCreated(BaseEvent[EventType.TimerCreated]):
    event_type: Literal[EventType.TimerCreated] = EventType.TimerCreated
    timer_id: TimerId
    delay_seconds: float


class TimerFired(BaseEvent[EventType.TimerFired]):
    event_type: Literal[EventType.TimerFired] = EventType.TimerFired
    timer_id: TimerId