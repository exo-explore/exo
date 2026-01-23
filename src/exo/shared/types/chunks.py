from collections.abc import Generator
from typing import Any, Literal

from exo.shared.models.model_cards import ModelId
from exo.shared.types.api import GenerationStats, ImageGenerationStats
from exo.utils.pydantic_ext import TaggedModel

from .api import FinishReason
from .common import CommandId
from .worker.runner_response import ToolCallItem


class BaseChunk(TaggedModel):
    model: ModelId


class TokenChunk(BaseChunk):
    text: str
    token_id: int
    finish_reason: Literal["stop", "length", "content_filter"] | None = None
    stats: GenerationStats | None = None


class ErrorChunk(BaseChunk):
    error_message: str
    finish_reason: Literal["error"] = "error"


class ToolCallChunk(BaseChunk):
    tool_calls: list[ToolCallItem]
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


class InputImageChunk(BaseChunk):
    command_id: CommandId
    data: str
    chunk_index: int
    total_chunks: int

    def __repr_args__(self) -> Generator[tuple[str, Any], None, None]:
        for name, value in super().__repr_args__():  # pyright: ignore[reportAny]
            if name == "data" and hasattr(value, "__len__"):  # pyright: ignore[reportAny]
                yield name, f"<{len(self.data)} chars>"
            elif name is not None:
                yield name, value


GenerationChunk = TokenChunk | ImageChunk | ToolCallChunk | ErrorChunk
