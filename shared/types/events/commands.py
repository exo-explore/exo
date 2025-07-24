from enum import Enum
from typing import Annotated, Callable, Literal, Sequence

from pydantic import BaseModel, Field, TypeAdapter

from shared.types.api import ChatCompletionTaskParams
from shared.types.events import CommandId, Event
from shared.types.state import InstanceId, State


# TODO: We need to have a distinction between create instance and spin up instance.
class CommandTypes(str, Enum):
    CHAT_COMPLETION = "CHAT_COMPLETION"
    CREATE_INSTANCE = "CREATE_INSTANCE"
    DELETE_INSTANCE = "DELETE_INSTANCE"


class _BaseCommand[T: CommandTypes](BaseModel):
    command_id: CommandId
    command_type: T


class ChatCompletionCommand(_BaseCommand[CommandTypes.CHAT_COMPLETION]):
    command_type: Literal[CommandTypes.CHAT_COMPLETION] = CommandTypes.CHAT_COMPLETION
    request_params: ChatCompletionTaskParams


class CreateInstanceCommand(_BaseCommand[CommandTypes.CREATE_INSTANCE]):
    command_type: Literal[CommandTypes.CREATE_INSTANCE] = CommandTypes.CREATE_INSTANCE
    model_id: str


class DeleteInstanceCommand(_BaseCommand[CommandTypes.DELETE_INSTANCE]):
    command_type: Literal[CommandTypes.DELETE_INSTANCE] = CommandTypes.DELETE_INSTANCE
    instance_id: InstanceId


Command = Annotated[
    ChatCompletionCommand | CreateInstanceCommand | DeleteInstanceCommand, 
    Field(discriminator="command_type")
]

CommandParser: TypeAdapter[Command] = TypeAdapter(Command)


type Decide = Callable[
    [State, Command],
    Sequence[Event],
]
