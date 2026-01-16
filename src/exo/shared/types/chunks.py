from enum import Enum
from typing import Literal

from pydantic import BaseModel

from exo.shared.types.api import GenerationStats
from exo.utils.pydantic_ext import TaggedModel

from .api import FinishReason
from .models import ModelId


class ChunkType(str, Enum):
    Token = "Token"
    Image = "Image"


class ToolCallFunction(BaseModel, frozen=True):
    name: str
    arguments: str


class ToolCall(BaseModel, frozen=True):
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class BaseChunk(TaggedModel):
    idx: int
    model: ModelId


class TokenChunk(BaseChunk):
    text: str
    token_id: int
    finish_reason: FinishReason | None = None
    stats: GenerationStats | None = None
    tool_calls: list[ToolCall] | None = None


class ImageChunk(BaseChunk):
    data: bytes


GenerationChunk = TokenChunk | ImageChunk
