from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from pydantic import BaseModel

from shared.types.common import NewUUID, NodeId


class EventId(NewUUID):
    """
    Newtype around `NewUUID`
    """


# Event base-class boilerplate (you should basically never touch these)
# Only very specialised registry or serialisation/deserialization logic might need know about these

class _EventType(str, Enum):
    """
    Here are all the unique kinds of events that can be sent over the network.
    """

    # Task Saga Events
    MLXInferenceSagaPrepare = "MLXInferenceSagaPrepare"
    MLXInferenceSagaStartPrepare = "MLXInferenceSagaStartPrepare"

    # Task Events
    TaskCreated = "TaskCreated"
    TaskStateUpdated = "TaskStateUpdated"
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

    # Node Performance Events
    NodePerformanceMeasured = "NodePerformanceMeasured"

    # Topology Events
    TopologyEdgeCreated = "TopologyEdgeCreated"
    TopologyEdgeReplacedAtomically = "TopologyEdgeReplacedAtomically"
    TopologyEdgeDeleted = "TopologyEdgeDeleted"
    WorkerConnected = "WorkerConnected"
    WorkerStatusUpdated = "WorkerStatusUpdated"
    WorkerDisconnected = "WorkerDisconnected"

    # # Timer Events
    # TimerCreated = "TimerCreated"
    # TimerFired = "TimerFired"


class _BaseEvent[T: _EventType](BaseModel):  # pyright: ignore[reportUnusedClass]
    """
    This is the event base-class, to please the Pydantic gods.
    PLEASE don't use this for anything unless you know why you are doing so,
    instead just use the events union :)
    """

    event_type: T
    event_id: EventId = EventId()

    def check_event_was_sent_by_correct_node(self, origin_id: NodeId) -> bool:
        """Check if the event was sent by the correct node.

        This is a placeholder implementation that always returns True.
        Subclasses can override this method to implement specific validation logic.
        """
        return True
