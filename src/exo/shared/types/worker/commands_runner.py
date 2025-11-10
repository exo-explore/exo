from exo.shared.openai_compat import FinishReason
from exo.utils.pydantic_ext import TaggedModel


class BaseRunnerResponse(TaggedModel):
    pass


class InitializedResponse(BaseRunnerResponse):
    time_taken: float


class TokenizedResponse(BaseRunnerResponse):
    prompt_tokens: int


class GenerationResponse(BaseRunnerResponse):
    text: str
    token: int
    # logprobs: list[float] | None = None # too big. we can change to be top-k
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
