from enum import Enum, auto
from typing import (
    Annotated,
    Callable,
    Generic,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
)

from pydantic import BaseModel, Field, TypeAdapter, model_validator

from shared.types.common import NewUUID, NodeId


class EventId(NewUUID):
    pass


class TimerId(NewUUID):
    pass


class MLXEventTypes(str, Enum):
    MLXInferenceSagaPrepare = "MLXInferenceSagaPrepare"
    MLXInferenceSagaStartPrepare = "MLXInferenceSagaStartPrepare"


class TaskEventTypes(str, Enum):
    TaskCreated = "TaskCreated"
    TaskUpdated = "TaskUpdated"
    TaskDeleted = "TaskDeleted"


class StreamingEventTypes(str, Enum):
    ChunkGenerated = "ChunkGenerated"


class InstanceEventTypes(str, Enum):
    InstanceCreated = "InstanceCreated"
    InstanceDeleted = "InstanceDeleted"
    InstanceToBeReplacedAtomically = "InstanceToBeReplacedAtomically"
    InstanceReplacedAtomically = "InstanceReplacedAtomically"
    InstanceStatusUpdated = "InstanceStatusUpdated"


class InstanceStateEventTypes(str, Enum):
    InstanceRunnerStateUpdated = "InstanceRunnerStateUpdated"


class NodePerformanceEventTypes(str, Enum):
    NodePerformanceProfiled = "NodePerformanceProfiled"


class DataPlaneEventTypes(str, Enum):
    DataPlaneEdgeCreated = "DataPlaneEdgeCreated"
    DataPlaneEdgeProfiled = "DataPlaneEdgeProfiled"
    DataPlaneEdgeDeleted = "DataPlaneEdgeDeleted"


class ControlPlaneEventTypes(str, Enum):
    WorkerConnected = "WorkerConnected"
    WorkerStatusUpdated = "WorkerStatusUpdated"
    WorkerDisconnected = "WorkerDisconnected"


class TimerEventTypes(str, Enum):
    TimerCreated = "TimerCreated"
    TimerFired = "TimerFired"


class ResourceEventTypes(str, Enum):
    ResourceProfiled = "ResourceProfiled"


class EventCategories(str, Enum):
    TaskEventTypes = auto()
    StreamingEventTypes = auto()
    InstanceEventTypes = auto()
    InstanceStateEventTypes = auto()
    NodePerformanceEventTypes = auto()
    ControlPlaneEventTypes = auto()
    DataPlaneEventTypes = auto()
    TimerEventTypes = auto()
    MLXEventTypes = auto()


PossibleEventOfEventTypeT = TypeVar("PossibleEventOfEventTypeT", bound=Enum)

#  T=(A|B) <: U=(A|B|C)  ==>  Event[A|B] <: Event[A|BCategoryOfEventsT_cov = TypeVar(name="CategoryOfEventsT_cov", bound=EventCategories, covariant=True)
CategoryOfEventsT_cov = TypeVar(
    name="CategoryOfEventsT_cov", bound=EventCategories, contravariant=True
)
CategoryOfEventsT_con = TypeVar(
    name="CategoryOfEventsT_con", bound=EventCategories, contravariant=True
)
CategoryOfEventsT_inv = TypeVar(
    name="CategoryOfEventsT_inv",
    bound=EventCategories,
    covariant=False,
    contravariant=False,
)


class Event(BaseModel, Generic[PossibleEventOfEventTypeT]):
    event_type: PossibleEventOfEventTypeT
    event_category: EventCategories
    event_id: EventId

    def check_origin_id(self, origin_id: NodeId) -> bool:
        return True


class TaskEvent(Event[TaskEventTypes]):
    event_type: TaskEventTypes


class InstanceEvent(Event[InstanceEventTypes]):
    event_type: InstanceEventTypes


class InstanceStateEvent(Event[InstanceStateEventTypes]):
    event_type: InstanceStateEventTypes


class MLXEvent(Event[MLXEventTypes]):
    event_type: MLXEventTypes


class NodePerformanceEvent(Event[NodePerformanceEventTypes]):
    event_type: NodePerformanceEventTypes


class ControlPlaneEvent(Event[ControlPlaneEventTypes]):
    event_type: ControlPlaneEventTypes


class StreamingEvent(Event[StreamingEventTypes]):
    event_type: StreamingEventTypes


class DataPlaneEvent(Event[DataPlaneEventTypes]):
    event_type: DataPlaneEventTypes


class TimerEvent(Event[TimerEventTypes]):
    event_type: TimerEventTypes


class ResourceEvent(Event[ResourceEventTypes]):
    event_type: ResourceEventTypes


class WrappedMessage(BaseModel, Generic[PossibleEventOfEventTypeT]):
    message: Event[PossibleEventOfEventTypeT]
    origin_id: NodeId

    @model_validator(mode="after")
    def check_origin_id(self) -> "WrappedMessage[PossibleEventOfEventTypeT]":
        if self.message.check_origin_id(self.origin_id):
            return self
        raise ValueError("Invalid Event: Origin ID Does Not Match")


class PersistedEvent(BaseModel, Generic[PossibleEventOfEventTypeT]):
    event: Event[PossibleEventOfEventTypeT]
    sequence_number: int = Field(gt=0)


class State(BaseModel, Generic[CategoryOfEventsT_cov]):
    event_category: CategoryOfEventsT_cov
    sequence_number: int = Field(default=0, ge=0)


AnnotatedEventType = Annotated[
    Event[EventCategories], Field(discriminator="event_category")
]
EventTypeParser: TypeAdapter[AnnotatedEventType] = TypeAdapter(AnnotatedEventType)


# it's not possible to enforce this at compile time, so we have to do it at runtime
def mock_todo[T](something: T | None) -> T: ...


def apply(
    state: State[CategoryOfEventsT_inv], event: Event[CategoryOfEventsT_inv]
) -> State[CategoryOfEventsT_inv]: ...


#  T=(A|B) <: U=(A|B|C)  ==>  Apply[A|B] <: Apply[A|B|C]
SagaApplicator = Callable[
    [State[CategoryOfEventsT_inv], Event[CategoryOfEventsT_inv]],
    Sequence[Event[CategoryOfEventsT_inv]],
]
Saga = Callable[
    [State[CategoryOfEventsT_inv], Event[CategoryOfEventsT_inv]],
    Sequence[Event[CategoryOfEventsT_inv]],
]
Apply = Callable[
    [State[CategoryOfEventsT_inv], Event[CategoryOfEventsT_inv]],
    State[CategoryOfEventsT_inv],
]
StateAndEvent = Tuple[State[CategoryOfEventsT_inv], Event[CategoryOfEventsT_inv]]
EffectHandler = Callable[
    [StateAndEvent[CategoryOfEventsT_inv], State[CategoryOfEventsT_inv]], None
]
EventPublisher = Callable[[Event[CategoryOfEventsT_inv]], None]


class MutableState[EventCategoryT: EventCategories](Protocol):
    def apply(
        self,
        event: Event[EventCategoryT],
        applicator: Apply[EventCategoryT],
        effect_handlers: Sequence[EffectHandler[EventCategoryT]],
    ) -> None: ...


class EventOutbox(Protocol):
    def send(self, events: Sequence[Event[EventCategories]]) -> None: ...


#
#  T=[A|B] <: U=[A|B|C]   =>   EventProcessor[A|B] :> EventProcessor[A|B|C]
#
class EventProcessor[EventCategoryT: EventCategories](Protocol):
    def get_events_to_apply(
        self, state: State[EventCategoryT]
    ) -> Sequence[Event[EventCategoryT]]: ...


def get_saga_effect_handler[EventCategoryT: EventCategories](
    saga: Saga[EventCategoryT], event_publisher: EventPublisher[EventCategoryT]
) -> EffectHandler[EventCategoryT]:
    def effect_handler(state_and_event: StateAndEvent[EventCategoryT]) -> None:
        trigger_state, trigger_event = state_and_event
        for event in saga(trigger_state, trigger_event):
            event_publisher(event)

    return lambda state_and_event, _: effect_handler(state_and_event)


def get_effects_from_sagas[EventCategoryT: EventCategories](
    sagas: Sequence[Saga[EventCategoryT]],
    event_publisher: EventPublisher[EventCategoryT],
) -> Sequence[EffectHandler[EventCategoryT]]:
    return [get_saga_effect_handler(saga, event_publisher) for saga in sagas]


IdemKeyGenerator = Callable[[State[CategoryOfEventsT_cov], int], Sequence[EventId]]


class CommandId(NewUUID):
    pass


class CommandTypes(str, Enum):
    Create = "Create"
    Update = "Update"
    Delete = "Delete"


class Command[EventCategoryT: EventCategories, CommandType: CommandTypes](BaseModel):
    command_type: CommandType
    command_id: CommandId


CommandTypeT = TypeVar("CommandTypeT", bound=CommandTypes, covariant=True)

Decide = Callable[
    [State[CategoryOfEventsT_cov], Command[CategoryOfEventsT_cov, CommandTypeT]],
    Sequence[Event[CategoryOfEventsT_cov]],
]
