from enum import Enum

from exo.shared.openai_compat import FinishReason
from exo.shared.types.models import ModelId
from exo.utils.pydantic_ext import TaggedModel


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


class ImageChunk(BaseChunk):
    data: bytes


GenerationChunk = TokenChunk | ImageChunk
