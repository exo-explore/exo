from typing import Any, Mapping, Type, get_args
from types import UnionType
from shared.constants import EXO_ERROR_REPORTING_MESSAGE
from shared.types.events.common import (
    Event,
    EventTypes,
    TaskEventTypes,
    InstanceEventTypes,
    NodePerformanceEventTypes,
    ControlPlaneEventTypes,
    StreamingEventTypes,
    DataPlaneEventTypes,
    MLXEventTypes,
    InstanceStateEventTypes,
)
from shared.types.events.events import (
    TaskCreated,
    TaskStateUpdated,
    TaskDeleted,
    InstanceCreated,
    InstanceDeleted,
    InstanceReplacedAtomically,
    InstanceSagaRunnerStateUpdated,
    NodePerformanceMeasured,
    WorkerConnected,
    WorkerStatusUpdated,
    WorkerDisconnected,
    ChunkGenerated,
    DataPlaneEdgeCreated,
    DataPlaneEdgeReplacedAtomically,
    DataPlaneEdgeDeleted,
    MLXInferenceSagaPrepare,
    MLXInferenceSagaStartPrepare,
)
from pydantic import TypeAdapter
from typing import Annotated
from pydantic import Field
from shared.types.events.common import EventCategories

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

EventRegistry: Mapping[EventTypes, Type[Any]] = {
    TaskEventTypes.TaskCreated: TaskCreated,
    TaskEventTypes.TaskStateUpdated: TaskStateUpdated,
    TaskEventTypes.TaskDeleted: TaskDeleted,
    InstanceEventTypes.InstanceCreated: InstanceCreated,
    InstanceEventTypes.InstanceDeleted: InstanceDeleted,
    InstanceEventTypes.InstanceReplacedAtomically: InstanceReplacedAtomically,
    InstanceStateEventTypes.InstanceSagaRunnerStateUpdated: InstanceSagaRunnerStateUpdated,
    NodePerformanceEventTypes.NodePerformanceMeasured: NodePerformanceMeasured,
    ControlPlaneEventTypes.WorkerConnected: WorkerConnected,
    ControlPlaneEventTypes.WorkerStatusUpdated: WorkerStatusUpdated,
    ControlPlaneEventTypes.WorkerDisconnected: WorkerDisconnected,
    StreamingEventTypes.ChunkGenerated: ChunkGenerated,
    DataPlaneEventTypes.DataPlaneEdgeCreated: DataPlaneEdgeCreated,
    DataPlaneEventTypes.DataPlaneEdgeReplacedAtomically: DataPlaneEdgeReplacedAtomically,
    DataPlaneEventTypes.DataPlaneEdgeDeleted: DataPlaneEdgeDeleted,
    MLXEventTypes.MLXInferenceSagaPrepare: MLXInferenceSagaPrepare,
    MLXEventTypes.MLXInferenceSagaStartPrepare: MLXInferenceSagaStartPrepare,
}


# Sanity Check.
def check_registry_has_all_event_types() -> None:
    event_types: tuple[EventTypes, ...] = get_args(EventTypes)
    missing_event_types = set(event_types) - set(EventRegistry.keys())

    assert not missing_event_types, (
        f"{EXO_ERROR_REPORTING_MESSAGE()}"
        f"There's an event missing from the registry: {missing_event_types}"
    )


def check_union_of_all_events_is_consistent_with_registry(
    registry: Mapping[EventTypes, Type[Any]], union_type: UnionType
) -> None:
    type_of_each_registry_entry = set(
        type(event_type) for event_type in registry.keys()
    )
    type_of_each_entry_in_union = set(get_args(union_type))
    missing_from_union = type_of_each_registry_entry - type_of_each_entry_in_union

    assert not missing_from_union, (
        f"{EXO_ERROR_REPORTING_MESSAGE()}"
        f"Event classes in registry are missing from all_events union: {missing_from_union}"
    )

    extra_in_union = type_of_each_entry_in_union - type_of_each_registry_entry

    assert not extra_in_union, (
        f"{EXO_ERROR_REPORTING_MESSAGE()}"
        f"Event classes in all_events union are missing from registry: {extra_in_union}"
    )


AllEvents = (
    TaskCreated
    | TaskStateUpdated
    | TaskDeleted
    | InstanceCreated
    | InstanceDeleted
    | InstanceReplacedAtomically
    | InstanceSagaRunnerStateUpdated
    | NodePerformanceMeasured
    | WorkerConnected
    | WorkerStatusUpdated
    | WorkerDisconnected
    | ChunkGenerated
    | DataPlaneEdgeCreated
    | DataPlaneEdgeReplacedAtomically
    | DataPlaneEdgeDeleted
    | MLXInferenceSagaPrepare
    | MLXInferenceSagaStartPrepare
)

# Run the sanity check
check_union_of_all_events_is_consistent_with_registry(EventRegistry, AllEvents)


_EventType = Annotated[AllEvents, Field(discriminator="event_type")]
EventParser: TypeAdapter[Event[EventCategories]] = TypeAdapter(_EventType)
