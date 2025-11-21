from datetime import datetime

from pydantic import Field

from exo.shared.topology import Connection, NodePerformanceProfile
from exo.shared.types.chunks import GenerationChunk
from exo.shared.types.common import CommandId, Id, NodeId, SessionId
from exo.shared.types.profiling import MemoryPerformanceProfile
from exo.shared.types.tasks import Task, TaskId, TaskStatus
from exo.shared.types.worker.downloads import DownloadProgress
from exo.shared.types.worker.instances import Instance, InstanceId
from exo.shared.types.worker.runners import RunnerId, RunnerStatus
from exo.utils.pydantic_ext import CamelCaseModel, TaggedModel


class EventId(Id):
    """
    Newtype around `ID`
    """


class BaseEvent(TaggedModel):
    event_id: EventId = Field(default_factory=EventId)
    # Internal, for debugging. Please don't rely on this field for anything!
    _master_time_stamp: None | datetime = None


class TestEvent(BaseEvent):
    pass


class TaskCreated(BaseEvent):
    task_id: TaskId
    task: Task


class TaskAcknowledged(BaseEvent):
    task_id: TaskId


class TaskDeleted(BaseEvent):
    task_id: TaskId


class TaskStatusUpdated(BaseEvent):
    task_id: TaskId
    task_status: TaskStatus


class TaskFailed(BaseEvent):
    task_id: TaskId
    error_type: str
    error_message: str


class InstanceCreated(BaseEvent):
    instance: Instance


class InstanceDeleted(BaseEvent):
    instance_id: InstanceId


class RunnerStatusUpdated(BaseEvent):
    runner_id: RunnerId
    runner_status: RunnerStatus


class RunnerDeleted(BaseEvent):
    runner_id: RunnerId


# TODO
class NodeCreated(BaseEvent):
    node_id: NodeId


class NodePerformanceMeasured(BaseEvent):
    node_id: NodeId
    node_profile: NodePerformanceProfile


class NodeDownloadProgress(BaseEvent):
    download_progress: DownloadProgress


class NodeMemoryMeasured(BaseEvent):
    node_id: NodeId
    memory: MemoryPerformanceProfile


class ChunkGenerated(BaseEvent):
    command_id: CommandId
    chunk: GenerationChunk


class TopologyEdgeCreated(BaseEvent):
    edge: Connection


class TopologyEdgeDeleted(BaseEvent):
    edge: Connection


Event = (
    TestEvent
    | TaskCreated
    | TaskStatusUpdated
    | TaskFailed
    | TaskDeleted
    | TaskAcknowledged
    | InstanceCreated
    | InstanceDeleted
    | RunnerStatusUpdated
    | RunnerDeleted
    | NodePerformanceMeasured
    | NodeMemoryMeasured
    | NodeDownloadProgress
    | ChunkGenerated
    | NodeCreated
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
