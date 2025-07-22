from enum import Enum
from typing import Annotated, Generic, Literal, TypeVar

from pydantic import BaseModel, Field, TypeAdapter

from shared.openai_compat import FinishReason
from shared.types.tasks import ChatCompletionTaskParams
from shared.types.worker.mlx import Host
from shared.types.worker.shards import ShardMetadata

## Messages passed TO the runner


class MessageType(str, Enum):
    Setup = "setup"
    ChatTask = "chat_task"
    Exit = "exit"


MT = TypeVar(name="MT", bound=MessageType)


class BaseRunnerMessage(BaseModel, Generic[MT]):
    pass


class SetupMessage(BaseRunnerMessage[MessageType.Setup]):
    type: Literal[MessageType.Setup] = Field(default=MessageType.Setup, frozen=True)
    model_shard_meta: ShardMetadata
    hosts: list[Host]


# TODO: We probably want a general task message that can take any task type. Can be fixed later.
class ChatTaskMessage(BaseRunnerMessage[MessageType.ChatTask]):
    type: Literal[MessageType.ChatTask] = Field(
        default=MessageType.ChatTask, frozen=True
    )
    task_data: ChatCompletionTaskParams


class ExitMessage(BaseRunnerMessage[MessageType.Exit]):
    type: Literal[MessageType.Exit] = Field(default=MessageType.Exit, frozen=True)


RunnerMessage = Annotated[
    SetupMessage | ChatTaskMessage | ExitMessage, Field(discriminator="type")
]
RunnerMessageTypeAdapter: TypeAdapter[RunnerMessage] = TypeAdapter(RunnerMessage)

## Responses passed FROM the runner


class RunnerResponseType(str, Enum):
    GenerationResponse = "generation_response"
    FinishedResponse = "finished_response"
    PrintResponse = "print_response"
    ErrorResponse = "error_response"


RRT = TypeVar(name="RRT", bound=RunnerResponseType)


class BaseRunnerResponse(BaseModel, Generic[RRT]):
    pass


class GenerationResponse(BaseRunnerResponse[RunnerResponseType.GenerationResponse]):
    type: Literal[RunnerResponseType.GenerationResponse] = Field(
        default=RunnerResponseType.GenerationResponse, frozen=True
    )
    text: str
    token: int
    # logprobs: Optional[list[float]] = None # too big. we can change to be top-k
    finish_reason: FinishReason | None = None


class PrintResponse(BaseRunnerResponse[RunnerResponseType.PrintResponse]):
    type: Literal[RunnerResponseType.PrintResponse] = Field(
        default=RunnerResponseType.PrintResponse, frozen=True
    )
    text: str


class FinishedResponse(BaseRunnerResponse[RunnerResponseType.FinishedResponse]):
    type: Literal[RunnerResponseType.FinishedResponse] = Field(
        default=RunnerResponseType.FinishedResponse, frozen=True
    )


class ErrorResponse(BaseRunnerResponse[RunnerResponseType.ErrorResponse]):
    type: Literal[RunnerResponseType.ErrorResponse] = Field(
        default=RunnerResponseType.ErrorResponse, frozen=True
    )
    error_type: str
    error_message: str
    traceback: str | None = None


RunnerResponse = Annotated[
    GenerationResponse | PrintResponse | FinishedResponse | ErrorResponse,
    Field(discriminator="type"),
]
RunnerResponseTypeAdapter: TypeAdapter[RunnerResponse] = TypeAdapter(RunnerResponse)
