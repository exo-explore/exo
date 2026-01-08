from enum import Enum

from exo.utils.pydantic_ext import TaggedModel

from .api import FinishReason
from .common import CommandId
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


class ImageChunk(BaseChunk):
    data: str
    chunk_index: int
    total_chunks: int
    image_index: int


class InputImageChunk(BaseChunk):
    command_id: CommandId
    data: str
    chunk_index: int
    total_chunks: int


GenerationChunk = TokenChunk | ImageChunk
