from pydantic import BaseModel

from exo.shared.types.api import (
    ChatCompletionTaskParams,
    CreateInstanceTaskParams,
    DeleteInstanceTaskParams,
)
from exo.shared.types.events import CommandId


class ChatCompletionCommand(BaseModel):
    command_id: CommandId
    command_params: ChatCompletionTaskParams


class CreateInstanceCommand(BaseModel):
    command_id: CommandId
    command_params: CreateInstanceTaskParams


class DeleteInstanceCommand(BaseModel):
    command_id: CommandId
    command_params: DeleteInstanceTaskParams


type Command = ChatCompletionCommand | CreateInstanceCommand | DeleteInstanceCommand
