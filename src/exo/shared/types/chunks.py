from enum import Enum

from exo.shared.models.model_cards import ModelId
from exo.shared.types.api import GenerationStats
from exo.utils.pydantic_ext import TaggedModel

from .api import FinishReason


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
    error_message: str | None = None


class ImageChunk(BaseChunk):
    data: str
    chunk_index: int
    total_chunks: int
    image_index: int


GenerationChunk = TokenChunk | ImageChunk
