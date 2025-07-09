from enum import Enum
from typing import Annotated, Generic, Literal, TypeVar

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from pydantic import BaseModel, Field, TypeAdapter

from shared.openai import FinishReason
from shared.types.models.common import ModelId
from shared.types.tasks.common import TaskId

OpenAIResponse = (
    ChatCompletion | ChatCompletionChunk
)  ## Currently we only support chat completions


class ChunkType(str, Enum):
    token = "token"
    image = "image"


ChunkT = TypeVar("ChunkT", bound=ChunkType)


class BaseChunk(BaseModel, Generic[ChunkT]):
    task_id: TaskId
    idx: int
    model: ModelId


###


class TokenChunkData(BaseModel):
    text: str
    token_id: int
    finish_reason: FinishReason | None = None


class ImageChunkData(BaseModel):
    data: bytes


###


class TokenChunk(BaseChunk[ChunkType.token]):
    chunk_data: TokenChunkData
    chunk_type: Literal[ChunkType.token] = Field(default=ChunkType.token, frozen=True)


class ImageChunk(BaseChunk[ChunkType.image]):
    chunk_data: ImageChunkData
    chunk_type: Literal[ChunkType.image] = Field(default=ChunkType.image, frozen=True)


###

GenerationChunk = Annotated[TokenChunk | ImageChunk, Field(discriminator="chunk_type")]
GenerationChunkTypeAdapter: TypeAdapter[GenerationChunk] = TypeAdapter(GenerationChunk)

# my_chunk: dict[str, Any] = TokenChunk(
#     task_id=TaskId('nicerid'),
#     idx=0,
#     chunk_data=TokenChunkData(
#         text='hello',
#         token_id=12,
#     ),
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
