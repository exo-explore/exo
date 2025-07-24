from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter

from shared.openai_compat import FinishReason
from shared.types.common import NewUUID
from shared.types.models import ModelId


class CommandId(NewUUID):
    """
    Newtype around `NewUUID` for command IDs
    """

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
GenerationChunkTypeAdapter: TypeAdapter[GenerationChunk] = TypeAdapter(GenerationChunk)

## OpenAIResponse = (
##    ChatCompletion | ChatCompletionChunk
## )  ## Currently we only support chat completions

# my_chunk: dict[str, Any] = TokenChunk(
#     task_id=TaskId('nicerid'),
#     idx=0,
    # text='hello',
    # token_id=12,
#     chunk_type=ChunkType.token,
#     model='llama-3.1',
# ).model_dump()
# print(my_chunk)
# restored = GenerationChunkTypeAdapter.validate_python(my_chunk)
# print(restored)

#### OpenAI API Interfaces ###

"""
def send_task(task: Any) -> AsyncGenerator[GenerationChunk]:
    # This is the 'command' - turns the task into an event and pushes to the event queue.
    # Tokens are then read off the event queue and pushed back to the api via an AsyncGenerator.
    ...

def parse_chunk_to_openai_response(chunk: GenerationChunk) -> OpenAIResponse:
    ...

async def handle_task(task: Any) -> AsyncGenerator[OpenAIResponse]:
    ## In our api call function, we will do:
    generator: AsyncGenerator[GenerationChunk] = send_task(task)

    async for chunk in generator:
        yield parse_chunk_to_openai_response(chunk)
"""
