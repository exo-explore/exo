from enum import Enum
from typing import Annotated, Callable, Literal, Sequence

from pydantic import BaseModel, Field, TypeAdapter

from exo.shared.types.api import ChatCompletionTaskParams
from exo.shared.types.common import CommandId
from exo.shared.types.events import Event
from exo.shared.types.models import ModelMetadata
from exo.shared.types.state import State
from exo.shared.types.worker.common import InstanceId


# TODO: We need to have a distinction between create instance and spin up instance.
class CommandType(str, Enum):
    CHAT_COMPLETION = "CHAT_COMPLETION"
    CREATE_INSTANCE = "CREATE_INSTANCE"
    DELETE_INSTANCE = "DELETE_INSTANCE"
    TASK_FINISHED = "TASK_FINISHED"


class _BaseCommand[T: CommandType](BaseModel):
    command_id: CommandId
    command_type: T


class ChatCompletionCommand(_BaseCommand[CommandType.CHAT_COMPLETION]):
    command_type: Literal[CommandType.CHAT_COMPLETION] = CommandType.CHAT_COMPLETION
    request_params: ChatCompletionTaskParams


class CreateInstanceCommand(_BaseCommand[CommandType.CREATE_INSTANCE]):
    command_type: Literal[CommandType.CREATE_INSTANCE] = CommandType.CREATE_INSTANCE
    model_meta: ModelMetadata
    instance_id: InstanceId


class DeleteInstanceCommand(_BaseCommand[CommandType.DELETE_INSTANCE]):
    command_type: Literal[CommandType.DELETE_INSTANCE] = CommandType.DELETE_INSTANCE
    instance_id: InstanceId


class TaskFinishedCommand(_BaseCommand[CommandType.TASK_FINISHED]):
    command_type: Literal[CommandType.TASK_FINISHED] = CommandType.TASK_FINISHED


Command = Annotated[
    ChatCompletionCommand | CreateInstanceCommand | DeleteInstanceCommand | TaskFinishedCommand, 
    Field(discriminator="command_type")
]

CommandParser: TypeAdapter[Command] = TypeAdapter(Command)


type Decide = Callable[
    [State, Command],
    Sequence[Event],
]
