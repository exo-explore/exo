from exo.shared.openai_compat import FinishReason
from exo.shared.types.common import Host
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.pydantic_ext import TaggedModel


class BaseRunnerMessage(TaggedModel):
    pass


class SetupMessage(BaseRunnerMessage):
    model_shard_meta: ShardMetadata
    hosts: list[Host] | None = None
    mlx_ibv_devices: list[list[str | None]] | None = None
    mlx_ibv_coordinator: str | None = None


# TODO: We probably want a general task message that can take any task type. Can be fixed later.
class ChatTaskMessage(BaseRunnerMessage):
    task_data: ChatCompletionTaskParams


class ExitMessage(BaseRunnerMessage):
    pass


RunnerMessage = SetupMessage | ChatTaskMessage | ExitMessage


class BaseRunnerResponse(TaggedModel):
    pass


class InitializedResponse(BaseRunnerResponse):
    time_taken: float


class TokenizedResponse(BaseRunnerResponse):
    prompt_tokens: int


class GenerationResponse(BaseRunnerResponse):
    text: str
    token: int
    # logprobs: Optional[list[float]] = None # too big. we can change to be top-k
    finish_reason: FinishReason | None = None


class PrintResponse(BaseRunnerResponse):
    text: str


class FinishedResponse(BaseRunnerResponse):
    pass


class ErrorResponse(BaseRunnerResponse):
    error_type: str
    error_message: str
    traceback: str


RunnerResponse = (
    InitializedResponse
    | TokenizedResponse
    | GenerationResponse
    | PrintResponse
    | FinishedResponse
    | ErrorResponse
)
