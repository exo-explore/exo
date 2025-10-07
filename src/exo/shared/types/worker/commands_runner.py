from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter

from exo.shared.openai_compat import FinishReason
from exo.shared.types.common import Host
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.shards import ShardMetadata


## Messages passed TO the runner
class MessageType(str, Enum):
    Setup = "setup"
    ChatTask = "chat_task"
    Exit = "exit"


class BaseRunnerMessage[MT: MessageType](BaseModel):
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
    InitializedResponse = "initialized_response"
    TokenizedResponse = "tokenized_response"
    GenerationResponse = "generation_response"
    FinishedResponse = "finished_response"
    PrintResponse = "print_response"
    ErrorResponse = "error_response"


class BaseRunnerResponse[RRT: RunnerResponseType](BaseModel):
    pass


class InitializedResponse(BaseRunnerResponse[RunnerResponseType.InitializedResponse]):
    type: Literal[RunnerResponseType.InitializedResponse] = Field(
        default=RunnerResponseType.InitializedResponse, frozen=True
    )
    time_taken: float


class TokenizedResponse(BaseRunnerResponse[RunnerResponseType.TokenizedResponse]):
    type: Literal[RunnerResponseType.TokenizedResponse] = Field(
        default=RunnerResponseType.TokenizedResponse, frozen=True
    )
    prompt_tokens: int


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
    traceback: str


RunnerResponse = Annotated[
    InitializedResponse
    | TokenizedResponse
    | GenerationResponse
    | PrintResponse
    | FinishedResponse
    | ErrorResponse,
    Field(discriminator="type"),
]
RunnerResponseTypeAdapter: TypeAdapter[RunnerResponse] = TypeAdapter(RunnerResponse)
