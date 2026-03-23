from collections.abc import Generator
from typing import Any, Literal, override
from uuid import uuid4

from pydantic import Field

from exo_core.models import CamelCaseModel, TaggedModel
from exo_core.utils.memory import Memory

FinishReason = Literal[
    "stop", "length", "tool_calls", "content_filter", "function_call", "error"
]


class ToolCallItem(CamelCaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    arguments: str


class TopLogprobItem(CamelCaseModel):
    token: str
    logprob: float
    bytes: list[int] | None = None


class GenerationStats(CamelCaseModel):
    prompt_tps: float
    generation_tps: float
    prompt_tokens: int
    generation_tokens: int
    peak_memory_usage: Memory


class ImageGenerationStats(CamelCaseModel):
    seconds_per_step: float
    total_generation_time: float

    num_inference_steps: int
    num_images: int

    image_width: int
    image_height: int

    peak_memory_usage: Memory


class PromptTokensDetails(CamelCaseModel):
    cached_tokens: int = 0
    audio_tokens: int = 0


class CompletionTokensDetails(CamelCaseModel):
    reasoning_tokens: int = 0
    audio_tokens: int = 0
    accepted_prediction_tokens: int = 0
    rejected_prediction_tokens: int = 0


class Usage(CamelCaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: PromptTokensDetails
    completion_tokens_details: CompletionTokensDetails


class BaseRunnerResponse(TaggedModel):
    pass


class TokenizedResponse(BaseRunnerResponse):
    prompt_tokens: int


class GenerationResponse(BaseRunnerResponse):
    text: str
    token: int
    logprob: float | None = None
    top_logprobs: list[TopLogprobItem] | None = None
    finish_reason: FinishReason | None = None
    stats: GenerationStats | None = None
    usage: Usage | None
    is_thinking: bool = False


class ImageGenerationResponse(BaseRunnerResponse):
    image_data: bytes
    format: Literal["png", "jpeg", "webp"] = "png"
    stats: ImageGenerationStats | None = None
    image_index: int = 0

    @override
    def __repr_args__(self) -> Generator[tuple[str, Any], None, None]:
        for name, value in super().__repr_args__():  # pyright: ignore[reportAny]
            if name == "image_data":
                yield name, f"<{len(self.image_data)} bytes>"
            elif name is not None:
                yield name, value


class PartialImageResponse(BaseRunnerResponse):
    image_data: bytes
    format: Literal["png", "jpeg", "webp"] = "png"
    partial_index: int
    total_partials: int
    image_index: int = 0

    @override
    def __repr_args__(self) -> Generator[tuple[str, Any], None, None]:
        for name, value in super().__repr_args__():  # pyright: ignore[reportAny]
            if name == "image_data":
                yield name, f"<{len(self.image_data)} bytes>"
            elif name is not None:
                yield name, value


class ToolCallResponse(BaseRunnerResponse):
    tool_calls: list[ToolCallItem]
    usage: Usage | None
    stats: GenerationStats | None = None


class FinishedResponse(BaseRunnerResponse):
    pass


class PrefillProgressResponse(BaseRunnerResponse):
    processed_tokens: int
    total_tokens: int
