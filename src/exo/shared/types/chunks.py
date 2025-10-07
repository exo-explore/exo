from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from exo.shared.openai_compat import FinishReason
from exo.shared.types.common import CommandId
from exo.shared.types.models import ModelId


class ChunkType(str, Enum):
    token = "token"
    image = "image"


class BaseChunk[ChunkTypeT: ChunkType](BaseModel):
    chunk_type: ChunkTypeT
    command_id: CommandId
    idx: int
    model: ModelId


class TokenChunk(BaseChunk[ChunkType.token]):
    chunk_type: Literal[ChunkType.token] = Field(default=ChunkType.token, frozen=True)
    text: str
    token_id: int
    finish_reason: FinishReason | None = None


class ImageChunk(BaseChunk[ChunkType.image]):
    chunk_type: Literal[ChunkType.image] = Field(default=ChunkType.image, frozen=True)
    data: bytes


GenerationChunk = Annotated[TokenChunk | ImageChunk, Field(discriminator="chunk_type")]
