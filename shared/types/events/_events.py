from typing import Annotated, Literal, Union

from pydantic import Field

from shared.topology import Connection, ConnectionProfile, Node, NodePerformanceProfile
from shared.types.common import NodeId
from shared.types.events import CommandId
from shared.types.events.chunks import GenerationChunk
from shared.types.tasks import Task, TaskId, TaskStatus
from shared.types.worker.common import InstanceId, NodeStatus
from shared.types.worker.instances import InstanceParams, TypeOfInstance
from shared.types.worker.runners import RunnerId, RunnerStatus

from ._common import _BaseEvent, _EventType  # pyright: ignore[reportPrivateUsage]


class TaskCreated(_BaseEvent[_EventType.TaskCreated]):
    event_type: Literal[_EventType.TaskCreated] = _EventType.TaskCreated
    task_id: TaskId
    task: Task


class TaskDeleted(_BaseEvent[_EventType.TaskDeleted]):
    event_type: Literal[_EventType.TaskDeleted] = _EventType.TaskDeleted
    task_id: TaskId


class TaskStateUpdated(_BaseEvent[_EventType.TaskStateUpdated]):
    event_type: Literal[_EventType.TaskStateUpdated] = _EventType.TaskStateUpdated
    task_id: TaskId
    task_status: TaskStatus


class InstanceCreated(_BaseEvent[_EventType.InstanceCreated]):
    event_type: Literal[_EventType.InstanceCreated] = _EventType.InstanceCreated
    instance_id: InstanceId
    instance_params: InstanceParams
    instance_type: TypeOfInstance


class InstanceActivated(_BaseEvent[_EventType.InstanceActivated]):
    event_type: Literal[_EventType.InstanceActivated] = _EventType.InstanceActivated
    instance_id: InstanceId


class InstanceDeactivated(_BaseEvent[_EventType.InstanceDeactivated]):
    event_type: Literal[_EventType.InstanceDeactivated] = _EventType.InstanceDeactivated
    instance_id: InstanceId


class InstanceDeleted(_BaseEvent[_EventType.InstanceDeleted]):
    event_type: Literal[_EventType.InstanceDeleted] = _EventType.InstanceDeleted
    instance_id: InstanceId

    transition: tuple[InstanceId, InstanceId]


class InstanceReplacedAtomically(_BaseEvent[_EventType.InstanceReplacedAtomically]):
    event_type: Literal[_EventType.InstanceReplacedAtomically] = _EventType.InstanceReplacedAtomically
    instance_to_replace: InstanceId
    new_instance_id: InstanceId


class RunnerStatusUpdated(_BaseEvent[_EventType.RunnerStatusUpdated]):
    event_type: Literal[_EventType.RunnerStatusUpdated] = _EventType.RunnerStatusUpdated
    runner_id: RunnerId
    runner_status: RunnerStatus


class MLXInferenceSagaPrepare(_BaseEvent[_EventType.MLXInferenceSagaPrepare]):
    event_type: Literal[_EventType.MLXInferenceSagaPrepare] = _EventType.MLXInferenceSagaPrepare
    task_id: TaskId
    instance_id: InstanceId


class MLXInferenceSagaStartPrepare(_BaseEvent[_EventType.MLXInferenceSagaStartPrepare]):
    event_type: Literal[_EventType.MLXInferenceSagaStartPrepare] = _EventType.MLXInferenceSagaStartPrepare
    task_id: TaskId
    instance_id: InstanceId


class NodePerformanceMeasured(_BaseEvent[_EventType.NodePerformanceMeasured]):
    event_type: Literal[_EventType.NodePerformanceMeasured] = _EventType.NodePerformanceMeasured
    node_id: NodeId
    node_profile: NodePerformanceProfile


class WorkerConnected(_BaseEvent[_EventType.WorkerConnected]):
    event_type: Literal[_EventType.WorkerConnected] = _EventType.WorkerConnected
    edge: Connection


class WorkerStatusUpdated(_BaseEvent[_EventType.WorkerStatusUpdated]):
    event_type: Literal[_EventType.WorkerStatusUpdated] = _EventType.WorkerStatusUpdated
    node_id: NodeId
    node_state: NodeStatus


class WorkerDisconnected(_BaseEvent[_EventType.WorkerDisconnected]):
    event_type: Literal[_EventType.WorkerDisconnected] = _EventType.WorkerDisconnected
    vertex_id: NodeId


class ChunkGenerated(_BaseEvent[_EventType.ChunkGenerated]):
    event_type: Literal[_EventType.ChunkGenerated] = _EventType.ChunkGenerated
    command_id: CommandId
    chunk: GenerationChunk


class TopologyEdgeCreated(_BaseEvent[_EventType.TopologyEdgeCreated]):
    event_type: Literal[_EventType.TopologyEdgeCreated] = _EventType.TopologyEdgeCreated
    vertex: Node


class TopologyEdgeReplacedAtomically(_BaseEvent[_EventType.TopologyEdgeReplacedAtomically]):
    event_type: Literal[_EventType.TopologyEdgeReplacedAtomically] = _EventType.TopologyEdgeReplacedAtomically
    edge: Connection
    edge_profile: ConnectionProfile


class TopologyEdgeDeleted(_BaseEvent[_EventType.TopologyEdgeDeleted]):
    event_type: Literal[_EventType.TopologyEdgeDeleted] = _EventType.TopologyEdgeDeleted
    edge: Connection

_Event = Union[
    TaskCreated,
    TaskStateUpdated,
    TaskDeleted,
    InstanceCreated,
    InstanceActivated,
    InstanceDeactivated,
    InstanceDeleted,
    InstanceReplacedAtomically,
    RunnerStatusUpdated,
    NodePerformanceMeasured,
    WorkerConnected,
    WorkerStatusUpdated,
    WorkerDisconnected,
    ChunkGenerated,
    TopologyEdgeCreated,
    TopologyEdgeReplacedAtomically,
    TopologyEdgeDeleted,
    MLXInferenceSagaPrepare,
    MLXInferenceSagaStartPrepare,
]
"""
Un-annotated union of all events. Only used internally to create the registry.
For all other usecases, use the annotated union of events :class:`Event` :)
"""

Event = Annotated[_Event, Field(discriminator="event_type")]
"""Type of events, a discriminated union."""

# class TimerCreated(_BaseEvent[_EventType.TimerCreated]):
#     event_type: Literal[_EventType.TimerCreated] = _EventType.TimerCreated
#     timer_id: TimerId
#     delay_seconds: float
#
#
# class TimerFired(_BaseEvent[_EventType.TimerFired]):
#     event_type: Literal[_EventType.TimerFired] = _EventType.TimerFired
#     timer_id: TimerId