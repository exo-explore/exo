from exo.shared.types.api import FinishReason, GenerationStats
from exo.utils.pydantic_ext import TaggedModel


class BaseRunnerResponse(TaggedModel):
    pass


class TokenizedResponse(BaseRunnerResponse):
    prompt_tokens: int


class GenerationResponse(BaseRunnerResponse):
    text: str
    token: int
    # logprobs: list[float] | None = None # too big. we can change to be top-k
    finish_reason: FinishReason | None = None
    stats: GenerationStats | None = None


class FinishedResponse(BaseRunnerResponse):
    pass
