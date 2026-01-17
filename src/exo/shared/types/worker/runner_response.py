from exo.shared.types.api import FinishReason, GenerationStats, TopLogprobItem
from exo.utils.pydantic_ext import TaggedModel


class BaseRunnerResponse(TaggedModel):
    pass


class TokenizedResponse(BaseRunnerResponse):
    prompt_tokens: int


class GenerationResponse(BaseRunnerResponse):
    text: str
    token: int
    logprob: float | None = None  # Log probability of the selected token
    top_logprobs: list[TopLogprobItem] | None = None  # Top-k alternative tokens
    finish_reason: FinishReason | None = None
    stats: GenerationStats | None = None


class FinishedResponse(BaseRunnerResponse):
    pass
