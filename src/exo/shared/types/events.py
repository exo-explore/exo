from datetime import datetime
from enum import Enum

from pydantic import Field

from exo.shared.topology import Connection, NodePerformanceProfile
from exo.shared.types.chunks import CommandId, GenerationChunk
from exo.shared.types.common import Id, NodeId, SessionId
from exo.shared.types.profiling import MemoryPerformanceProfile
from exo.shared.types.tasks import Task, TaskId, TaskStatus
from exo.shared.types.worker.common import InstanceId, WorkerStatus
from exo.shared.types.worker.instances import Instance
from exo.shared.types.worker.runners import RunnerId, RunnerStatus
from exo.utils.pydantic_ext import CamelCaseModel, TaggedModel


class EventId(Id):
    """
    Newtype around `ID`
    """


class EventType(str, Enum):
    """
    Here are all the unique kinds of events that can be sent over the network.
    """

    # Test Events, strictly for mocks and tests.
    TestEvent = "TestEvent"

    # Task Events
    TaskCreated = "TaskCreated"
    TaskStateUpdated = "TaskStateUpdated"
    TaskFailed = "TaskFailed"
    TaskDeleted = "TaskDeleted"

    # Streaming Events
    ChunkGenerated = "ChunkGenerated"

    # Instance Events
    InstanceCreated = "InstanceCreated"
    InstanceDeleted = "InstanceDeleted"
    InstanceActivated = "InstanceActivated"
    InstanceDeactivated = "InstanceDeactivated"
    InstanceReplacedAtomically = "InstanceReplacedAtomically"

    # Runner Status Events
    RunnerStatusUpdated = "RunnerStatusUpdated"
    RunnerDeleted = "RunnerDeleted"

    # Node Performance Events
    WorkerStatusUpdated = "WorkerStatusUpdated"
    NodePerformanceMeasured = "NodePerformanceMeasured"
    NodeMemoryMeasured = "NodeMemoryMeasured"

    # Topology Events
    TopologyNodeCreated = "TopologyNodeCreated"
    TopologyEdgeCreated = "TopologyEdgeCreated"
    TopologyEdgeDeleted = "TopologyEdgeDeleted"


class BaseEvent(TaggedModel):
    event_id: EventId = Field(default_factory=EventId)
    # Internal, for debugging. Please don't rely on this field for anything!
    _master_time_stamp: None | datetime = None


class TestEvent(BaseEvent):
    pass


class TaskCreated(BaseEvent):
    task_id: TaskId
    task: Task


class TaskDeleted(BaseEvent):
    task_id: TaskId


class TaskStateUpdated(BaseEvent):
    task_id: TaskId
    task_status: TaskStatus


class TaskFailed(BaseEvent):
    task_id: TaskId
    error_type: str
    error_message: str


class InstanceCreated(BaseEvent):
    instance: Instance


class InstanceActivated(BaseEvent):
    instance_id: InstanceId


class InstanceDeactivated(BaseEvent):
    instance_id: InstanceId


class InstanceDeleted(BaseEvent):
    instance_id: InstanceId


class RunnerStatusUpdated(BaseEvent):
    runner_id: RunnerId
    runner_status: RunnerStatus


class RunnerDeleted(BaseEvent):
    runner_id: RunnerId


class NodePerformanceMeasured(BaseEvent):
    node_id: NodeId
    node_profile: NodePerformanceProfile


class NodeMemoryMeasured(BaseEvent):
    node_id: NodeId
    memory: MemoryPerformanceProfile


class WorkerStatusUpdated(BaseEvent):
    node_id: NodeId
    node_state: WorkerStatus


class ChunkGenerated(BaseEvent):
    command_id: CommandId
    chunk: GenerationChunk


class TopologyNodeCreated(BaseEvent):
    node_id: NodeId


class TopologyEdgeCreated(BaseEvent):
    edge: Connection


class TopologyEdgeDeleted(BaseEvent):
    edge: Connection


Event = (
    TestEvent
    | TaskCreated
    | TaskStateUpdated
    | TaskFailed
    | TaskDeleted
    | InstanceCreated
    | InstanceActivated
    | InstanceDeactivated
    | InstanceDeleted
    | RunnerStatusUpdated
    | RunnerDeleted
    | NodePerformanceMeasured
    | NodeMemoryMeasured
    | WorkerStatusUpdated
    | ChunkGenerated
    | TopologyNodeCreated
    | TopologyEdgeCreated
    | TopologyEdgeDeleted
)


class IndexedEvent(CamelCaseModel):
    """An event indexed by the master, with a globally unique index"""

    idx: int = Field(ge=0)
    event: Event


class ForwarderEvent(CamelCaseModel):
    """An event the forwarder will serialize and send over the network"""

    origin_idx: int = Field(ge=0)
    origin: NodeId
    session: SessionId
    event: Event
