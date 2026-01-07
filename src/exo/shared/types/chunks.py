from enum import Enum

from exo.shared.types.api import GenerationStats
from exo.utils.pydantic_ext import TaggedModel

from .api import FinishReason
from .models import ModelId


class ChunkType(str, Enum):
    Token = "Token"
    Image = "Image"


class BaseChunk(TaggedModel):
    idx: int
    model: ModelId


class TokenChunk(BaseChunk):
    text: str
    token_id: int
    finish_reason: FinishReason | None = None
    stats: GenerationStats | None = None


class ImageChunk(BaseChunk):
    data: bytes


GenerationChunk = TokenChunk | ImageChunk
