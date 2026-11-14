from collections.abc import Generator
from typing import Annotated, Any, Literal

from pydantic import model_validator

from exo.api.types import (
    FinishReason,
    GenerationStats,
    ImageGenerationStats,
    ToolCallItem,
    TopLogprobItem,
    Usage,
)
from exo.api.types.api import LogprobsContentItem
from exo.shared.models.model_cards import ModelId
from exo.utils.pydantic_ext import ByTag, TaggedModel
from exo.worker.runner.diagnostics import KnownRunnerDiagnostic


class BaseChunk(TaggedModel):
    model: ModelId


class TokenChunk(BaseChunk):
    text: str
    token_id: int | None
    usage: Usage | None
    finish_reason: Literal["stop", "length", "content_filter"] | None = None
    stats: GenerationStats | None = None
    logprob: float | None = None
    top_logprobs: list[TopLogprobItem] | None = None
    logprobs_content: list[LogprobsContentItem] | None = None
    is_thinking: bool = False

    @model_validator(mode="after")
    def validate_token_identity(self) -> "TokenChunk":
        if self.token_id is None and (
            self.text
            or self.finish_reason is None
            or self.usage is None
            or self.usage.completion_tokens != 0
            or self.logprob is not None
            or self.top_logprobs
            or self.logprobs_content
        ):
            raise ValueError(
                "Only an empty zero-token completion may omit its token ID"
            )
        return self


class ErrorChunk(BaseChunk):
    # One short sentence for the user (see exo.shared.request_errors); details go to the logs
    error_message: str
    finish_reason: Literal["error"] = "error"

    # NOTE: this is a bad place to put this, creates semantic overlap/confusion;
    #       at some point someone put this somewhere else, thanks :)
    diagnostics: list[KnownRunnerDiagnostic] = []


class ToolCallChunk(BaseChunk):
    tool_calls: list[ToolCallItem]
    usage: Usage | None
    finish_reason: Literal["tool_calls"] = "tool_calls"
    stats: GenerationStats | None = None


class ImageChunk(BaseChunk):
    data: str
    chunk_index: int
    total_chunks: int
    image_index: int
    is_partial: bool = False
    partial_index: int | None = None
    total_partials: int | None = None
    stats: ImageGenerationStats | None = None
    format: Literal["png", "jpeg", "webp"] | None = None
    finish_reason: FinishReason | None = None
    error_message: str | None = None

    def __repr_args__(self) -> Generator[tuple[str, Any], None, None]:
        for name, value in super().__repr_args__():  # pyright: ignore[reportAny]
            if name == "data" and hasattr(value, "__len__"):  # pyright: ignore[reportAny]
                yield name, f"<{len(self.data)} chars>"
            elif name is not None:
                yield name, value


class PrefillProgressChunk(BaseChunk):
    """Data class for prefill progress events during streaming."""

    processed_tokens: int
    total_tokens: int


class EngineHeartbeatChunk(BaseChunk):
    pass


StatusChunk = PrefillProgressChunk | EngineHeartbeatChunk
GenerationChunk = TokenChunk | ImageChunk | ToolCallChunk | ErrorChunk
# Validated by tag (see ByTag): every token of a reply is one
Chunk = Annotated[StatusChunk | GenerationChunk, ByTag()]
