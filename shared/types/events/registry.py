from typing import Annotated, Any, Mapping, Type, TypeAlias

from pydantic import Field, TypeAdapter

from shared.types.events.common import (
    EventType,
)
from shared.types.events.events import (
    ChunkGenerated,
    InstanceActivated,
    InstanceCreated,
    InstanceDeactivated,
    InstanceDeleted,
    InstanceReplacedAtomically,
    MLXInferenceSagaPrepare,
    MLXInferenceSagaStartPrepare,
    NodePerformanceMeasured,
    RunnerStatusUpdated,
    TaskCreated,
    TaskDeleted,
    TaskStateUpdated,
    TimerCreated,
    TimerFired,
    TopologyEdgeCreated,
    TopologyEdgeDeleted,
    TopologyEdgeReplacedAtomically,
    WorkerConnected,
    WorkerDisconnected,
    WorkerStatusUpdated,
)
from shared.types.events.sanity_checking import (
    assert_event_union_covers_registry,
    check_registry_has_all_event_types,
    check_union_of_all_events_is_consistent_with_registry,
)

"""
class EventTypeNames(StrEnum):
    TaskEventType = auto()
    InstanceEvent = auto()
    NodePerformanceEvent = auto()
    ControlPlaneEvent = auto()
    StreamingEvent = auto()
    DataPlaneEvent = auto()
    TimerEvent = auto()
    MLXEvent = auto()

check_event_categories_are_defined_for_all_event_types(EVENT_TYPE_ENUMS, EventTypeNames)
"""
EventRegistry: Mapping[EventType, Type[Any]] = {
    EventType.TaskCreated: TaskCreated,
    EventType.TaskStateUpdated: TaskStateUpdated,
    EventType.TaskDeleted: TaskDeleted,
    EventType.InstanceCreated: InstanceCreated,
    EventType.InstanceActivated: InstanceActivated,
    EventType.InstanceDeactivated: InstanceDeactivated,
    EventType.InstanceDeleted: InstanceDeleted,
    EventType.InstanceReplacedAtomically: InstanceReplacedAtomically,
    EventType.RunnerStatusUpdated: RunnerStatusUpdated,
    EventType.NodePerformanceMeasured: NodePerformanceMeasured,
    EventType.WorkerConnected: WorkerConnected,
    EventType.WorkerStatusUpdated: WorkerStatusUpdated,
    EventType.WorkerDisconnected: WorkerDisconnected,
    EventType.ChunkGenerated: ChunkGenerated,
    EventType.TopologyEdgeCreated: TopologyEdgeCreated,
    EventType.TopologyEdgeReplacedAtomically: TopologyEdgeReplacedAtomically,
    EventType.TopologyEdgeDeleted: TopologyEdgeDeleted,
    EventType.MLXInferenceSagaPrepare: MLXInferenceSagaPrepare,
    EventType.MLXInferenceSagaStartPrepare: MLXInferenceSagaStartPrepare,
    EventType.TimerCreated: TimerCreated,
    EventType.TimerFired: TimerFired,
}


AllEventsUnion = (
    TaskCreated
    | TaskStateUpdated
    | TaskDeleted
    | InstanceCreated
    | InstanceActivated
    | InstanceDeactivated
    | InstanceDeleted
    | InstanceReplacedAtomically
    | RunnerStatusUpdated
    | NodePerformanceMeasured
    | WorkerConnected
    | WorkerStatusUpdated
    | WorkerDisconnected
    | ChunkGenerated
    | TopologyEdgeCreated
    | TopologyEdgeReplacedAtomically
    | TopologyEdgeDeleted
    | MLXInferenceSagaPrepare
    | MLXInferenceSagaStartPrepare
    | TimerCreated
    | TimerFired
)

Event: TypeAlias = Annotated[AllEventsUnion, Field(discriminator="event_type")]
EventParser: TypeAdapter[Event] = TypeAdapter(Event)




assert_event_union_covers_registry(AllEventsUnion)
check_union_of_all_events_is_consistent_with_registry(EventRegistry, AllEventsUnion)
check_registry_has_all_event_types(EventRegistry)