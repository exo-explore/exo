from enum import Enum
from typing import (
    TYPE_CHECKING,
    Generic,
    TypeVar,
)

if TYPE_CHECKING:
    pass

from pydantic import BaseModel

from shared.types.common import NewUUID, NodeId


class EventId(NewUUID):
    pass


class TimerId(NewUUID):
    pass


# Here are all the unique kinds of events that can be sent over the network.
class EventType(str, Enum):
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
    
    # Timer Events
    TimerCreated = "TimerCreated"
    TimerFired = "TimerFired"

EventTypeT = TypeVar("EventTypeT", bound=EventType)


class BaseEvent(BaseModel, Generic[EventTypeT]):
    event_type: EventTypeT
    event_id: EventId = EventId()

    def check_event_was_sent_by_correct_node(self, origin_id: NodeId) -> bool:
        """Check if the event was sent by the correct node.
        
        This is a placeholder implementation that always returns True.
        Subclasses can override this method to implement specific validation logic.
        """
        return True



