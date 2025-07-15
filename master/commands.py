from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter


class BaseExternalCommand[T: str](BaseModel):
    command_type: T


class ChatCompletionNonStreamingCommand(
    BaseExternalCommand[Literal["chat_completion_non_streaming"]]
):
    command_type: Literal["chat_completion_non_streaming"] = (
        "chat_completion_non_streaming"
    )


ExternalCommand = Annotated[
    ChatCompletionNonStreamingCommand, Field(discriminator="command_type")
]
ExternalCommandParser: TypeAdapter[ExternalCommand] = TypeAdapter(ExternalCommand)
