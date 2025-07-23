from enum import Enum
from typing import Annotated, Callable, Sequence

from pydantic import BaseModel, Field, TypeAdapter

from shared.types.api import ChatCompletionTaskParams
from shared.types.common import NewUUID
from shared.types.events import Event
from shared.types.state import InstanceId, State


class CommandId(NewUUID):
    pass


class CommandTypes(str, Enum):
    CHAT_COMPLETION = "CHAT_COMPLETION"
    CREATE_INSTANCE = "CREATE_INSTANCE"
    DELETE_INSTANCE = "DELETE_INSTANCE"


class _BaseCommand[T: CommandTypes](BaseModel):
    command_id: CommandId
    command_type: T


class ChatCompletionCommand(_BaseCommand[CommandTypes.CHAT_COMPLETION]):
    request_params: ChatCompletionTaskParams


class CreateInstanceCommand(_BaseCommand[CommandTypes.CREATE_INSTANCE]):
    model_id: str


class DeleteInstanceCommand(_BaseCommand[CommandTypes.DELETE_INSTANCE]):
    instance_id: InstanceId


Command = Annotated[
    ChatCompletionCommand, Field(discriminator="command_type")
]

CommandParser: TypeAdapter[Command] = TypeAdapter(Command)


type Decide = Callable[
    [State, Command],
    Sequence[Event],
]
