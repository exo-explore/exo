from typing import (
    Annotated,
    Callable,
    Generic,
    Literal,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    get_args,
)
from uuid import UUID

from pydantic import BaseModel, Field, TypeAdapter
from pydantic.types import UuidVersion

_EventId = Annotated[UUID, UuidVersion(4)]
EventId = type("EventId", (UUID,), {})
EventIdParser: TypeAdapter[EventId] = TypeAdapter(_EventId)

EventTypes = Literal[
    "ChatCompletionsRequestStarted",
    "ChatCompletionsRequestCompleted",
    "ChatCompletionsRequestFailed",
    "InferenceSagaStarted",
    "InferencePrepareStarted",
    "InferencePrepareCompleted",
    "InferenceTriggerStarted",
    "InferenceTriggerCompleted",
    "InferenceCompleted",
    "InferenceSagaCompleted",
    "InstanceSetupSagaStarted",
    "InstanceSetupSagaCompleted",
    "InstanceSetupSagaFailed",
    "ShardAssigned",
    "ShardAssignFailed",
    "ShardUnassigned",
    "ShardUnassignFailed",
    "ShardKilled",
    "ShardDied",
    "ShardSpawned",
    "ShardSpawnedFailed",
    "ShardDespawned",
    "NodeConnected",
    "NodeConnectionProfiled",
    "NodeDisconnected",
    "NodeStarted",
    "DeviceRegistered",
    "DeviceProfiled",
    "TokenGenerated",
    "RepoProgressEvent",
    "TimerScheduled",
    "TimerFired",
]
EventTypeT = TypeVar("EventTypeT", bound=EventTypes)
TEventType = TypeVar("TEventType", bound=EventTypes, covariant=True)


class Event(BaseModel, Generic[TEventType]):
    event_type: TEventType
    event_id: EventId


class PersistedEvent(BaseModel, Generic[TEventType]):
    event: Event[TEventType]
    sequence_number: int = Field(gt=0)


class State(BaseModel, Generic[EventTypeT]):
    event_types: tuple[EventTypeT, ...] = get_args(EventTypeT)
    sequence_number: int = Field(default=0, ge=0)


AnnotatedEventType = Annotated[EventTypes, Field(discriminator="event_type")]
EventTypeParser: TypeAdapter[AnnotatedEventType] = TypeAdapter(AnnotatedEventType)

Applicator = Callable[[State[EventTypeT], Event[TEventType]], State[EventTypeT]]
Apply = Callable[[State[EventTypeT], Event[EventTypeT]], State[EventTypeT]]
SagaApplicator = Callable[
    [State[EventTypeT], Event[TEventType]], Sequence[Event[EventTypeT]]
]
Saga = Callable[[State[EventTypeT], Event[EventTypeT]], Sequence[Event[EventTypeT]]]

StateAndEvent = Tuple[State[EventTypeT], Event[EventTypeT]]
EffectHandler = Callable[[StateAndEvent[EventTypeT], State[EventTypeT]], None]
EventPublisher = Callable[[Event[EventTypeT]], None]


class EventOutbox(Protocol):
    def send(self, events: Sequence[Event[EventTypeT]]) -> None: ...


class EventProcessor(Protocol):
    def update(
        self,
        state: State[EventTypeT],
        apply: Apply[EventTypeT],
        effect_handlers: Sequence[EffectHandler[EventTypeT]],
    ) -> State[EventTypeT]: ...


def get_saga_effect_handler(
    sagas: Saga[EventTypeT], event_publisher: EventPublisher[EventTypeT]
) -> EffectHandler[EventTypeT]:
    def effect_handler(state_and_event: StateAndEvent[EventTypeT]) -> None:
        trigger_state, trigger_event = state_and_event
        for event in sagas(trigger_state, trigger_event):
            event_publisher(event)

    return lambda state_and_event, _: effect_handler(state_and_event)


def get_effects_from_sagas(
    sagas: Sequence[Saga[EventTypeT]], event_publisher: EventPublisher[EventTypeT]
) -> Sequence[EffectHandler[EventTypeT]]:
    return [get_saga_effect_handler(saga, event_publisher) for saga in sagas]


IdemKeyGenerator = Callable[[State[EventTypeT], int], Sequence[EventId]]

_CommandId = Annotated[UUID, UuidVersion(4)]
CommandId = type("CommandId", (UUID,), {})
CommandIdParser: TypeAdapter[CommandId] = TypeAdapter(_CommandId)

CommandTypes = Literal["create", "update", "delete"]
CommandTypeT = TypeVar("CommandTypeT", bound=EventTypes)
TCommandType = TypeVar("TCommandType", bound=EventTypes, covariant=True)


class Command(BaseModel, Generic[TEventType, TCommandType]):
    command_type: TCommandType
    command_id: CommandId


Decide = Callable[
    [State[EventTypeT], Command[TEventType, TCommandType]], Sequence[Event[EventTypeT]]
]
