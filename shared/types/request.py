from pydantic import BaseModel

from shared.types.api import (
    ChatCompletionTaskParams,
    DeleteInstanceTaskParams,
    RequestInstanceTaskParams,
)
from shared.types.events import CommandId


class ChatCompletionCommand(BaseModel):
    command_id: CommandId
    command_params: ChatCompletionTaskParams

class RequestInstanceCommand(BaseModel):
    command_id: CommandId
    command_params: RequestInstanceTaskParams

class DeleteInstanceCommand(BaseModel):
    command_id: CommandId
    command_params: DeleteInstanceTaskParams

type Command = ChatCompletionCommand | RequestInstanceCommand | DeleteInstanceCommand
