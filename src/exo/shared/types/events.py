from datetime import datetime
from typing import Annotated, final

from pydantic import Field

from exo.shared.models.model_cards import ModelCard
from exo.shared.topology import Connection
from exo.shared.types.chunks import Chunk, InputImageChunk
from exo.shared.types.common import (
    CommandId,
    DeploymentId,
    Id,
    ModelId,
    NodeId,
    SessionId,
    SystemId,
)
from exo.shared.types.deployments import Deployment, Hold, Waiting
from exo.shared.types.instance_link import InstanceLink, InstanceLinkId
from exo.shared.types.state import State
from exo.shared.types.tasks import Task, TaskId, TaskStatus
from exo.shared.types.worker.downloads import DownloadProgress
from exo.shared.types.worker.instances import (
    Instance,
    InstanceId,
    InstanceRemovalReason,
)
from exo.shared.types.worker.runners import RunnerId, RunnerStatus
from exo.utils.info_gatherer.info_gatherer import GatheredInfo
from exo.utils.pydantic_ext import ByTag, FrozenModel, TaggedModel


class EventId(Id):
    """
    Newtype around `ID`
    """


class BaseEvent(TaggedModel):
    event_id: EventId = Field(default_factory=EventId)
    # Internal, for debugging. Please don't rely on this field for anything!
    _master_time_stamp: None | datetime = None


class TestEvent(BaseEvent):
    __test__ = False


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

    def __eq__(self, other: object) -> bool:
        if isinstance(other, InstanceCreated):
            return self.instance == other.instance and self.event_id == other.event_id

        return False


class InstanceDeleted(BaseEvent):
    instance_id: InstanceId
    reason: InstanceRemovalReason
    # When, as an ISO 8601 time (a datetime field in an event can't be read back from JSON)
    at: str


class DeploymentCreated(BaseEvent):
    """A deployment launched, or one a node knew of that the master didn't (after a change of
    master or a heal). For one the state has, only a higher revision changes it."""

    deployment: Deployment


class DeploymentRemoved(BaseEvent):
    """A user stopped it. Remembered as a tombstone, so it doesn't come back."""

    deployment_id: DeploymentId
    # ISO 8601 time of the stop
    at: str


class DeploymentHeld(BaseEvent):
    """It isn't placed again until the user resumes it."""

    deployment_id: DeploymentId
    hold: Hold


class DeploymentResumed(BaseEvent):
    """Placed now: whatever held it or made it wait is cleared. The user asked for it, or the
    master lifted its own hold (a node showed it running elsewhere, or not a full restart)."""

    deployment_id: DeploymentId
    # ISO time: how long it then goes without an instance counts from here
    at: str
    # Its pinned nodes aren't all here: the user asked to load it on the nodes that are
    unpin: bool = False
    # A user's resume is a change of the deployment (its revision); the master's isn't
    by_user: bool = True


class DeploymentStatus(BaseEvent):
    """Why it can't be placed now, and how it runs. Sent only when one of them changes."""

    deployment_id: DeploymentId
    waiting: Waiting | None
    note: str | None


class RunnerStatusUpdated(BaseEvent):
    runner_id: RunnerId
    runner_status: RunnerStatus


class NodeTimedOut(BaseEvent):
    node_id: NodeId


# TODO: bikeshed this name
class NodeGatheredInfo(BaseEvent):
    node_id: NodeId
    when: str  # this is a manually cast datetime overrode by the master when the event is indexed, rather than the local time on the device
    info: GatheredInfo


class NodeDownloadProgress(BaseEvent):
    download_progress: DownloadProgress


class ChunkGenerated(BaseEvent):
    command_id: CommandId
    chunk: Chunk


class InputChunkReceived(BaseEvent):
    command_id: CommandId
    chunk: InputImageChunk


class TopologyEdgeCreated(BaseEvent):
    conn: Connection


class TopologyEdgeDeleted(BaseEvent):
    conn: Connection


class CustomModelCardAdded(BaseEvent):
    model_card: ModelCard


class CustomModelCardDeleted(BaseEvent):
    model_id: ModelId


@final
class TraceEventData(FrozenModel):
    name: str
    start_us: int
    duration_us: int
    rank: int
    category: str


@final
class TracesCollected(BaseEvent):
    task_id: TaskId
    rank: int
    traces: list[TraceEventData]


@final
class TracesMerged(BaseEvent):
    task_id: TaskId
    traces: list[TraceEventData]


class InstanceLinkCreated(BaseEvent):
    link: InstanceLink


class InstanceLinkDeleted(BaseEvent):
    link_id: InstanceLinkId


# Validated by tag (see ByTag): a follower parses every event of the cluster, one per token
# of every reply
Event = Annotated[
    TestEvent
    | TaskCreated
    | TaskStatusUpdated
    | TaskFailed
    | TaskDeleted
    | TaskAcknowledged
    | InstanceCreated
    | InstanceDeleted
    | DeploymentCreated
    | DeploymentRemoved
    | DeploymentHeld
    | DeploymentResumed
    | DeploymentStatus
    | RunnerStatusUpdated
    | NodeTimedOut
    | NodeGatheredInfo
    | NodeDownloadProgress
    | ChunkGenerated
    | InputChunkReceived
    | TopologyEdgeCreated
    | TopologyEdgeDeleted
    | TracesCollected
    | TracesMerged
    | CustomModelCardAdded
    | CustomModelCardDeleted
    | InstanceLinkCreated
    | InstanceLinkDeleted,
    ByTag(),
]


class IndexedEvent(FrozenModel):
    """An event indexed by the master, with a globally unique index"""

    idx: int = Field(ge=0)
    event: Event


class GlobalForwarderEvent(FrozenModel):
    """An event the forwarder will serialize and send over the network"""

    origin_idx: int = Field(ge=0)
    origin: NodeId
    session: SessionId
    event: Event


class StateSnapshot(FrozenModel):
    """The master's state, sent to a node too far behind to replay the events it missed.

    ``state.last_event_applied_idx`` is the index of the last event it includes.
    """

    session: SessionId
    requester: SystemId
    state: State


class LocalForwarderEvent(FrozenModel):
    """An event the forwarder will serialize and send over the network"""

    origin_idx: int = Field(ge=0)
    origin: SystemId
    session: SessionId
    event: Event
