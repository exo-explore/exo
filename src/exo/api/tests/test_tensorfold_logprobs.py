from collections.abc import AsyncGenerator

from exo.api.adapters.chat_completions import chunk_to_response, collect_chat_response
from exo.api.types.api import (
    ChatCompletionChoice,
    ChatCompletionResponse,
    LogprobsContentItem,
    StreamingChoiceResponse,
)
from exo.shared.types.chunks import TokenChunk
from exo.shared.types.common import CommandId, ModelId


def grouped_chunk() -> TokenChunk:
    return TokenChunk(
        model=ModelId("test/native"),
        token_id=11,
        text="é",
        usage=None,
        finish_reason="stop",
        logprobs_content=[
            LogprobsContentItem(token="�", bytes=[195], logprob=-0.1, top_logprobs=[]),
            LogprobsContentItem(token="�", bytes=[169], logprob=-0.2, top_logprobs=[]),
        ],
    )


def test_streamed_logprobs_preserve_token_bytes_separately_from_content() -> None:
    chunk = grouped_chunk()
    response = chunk_to_response(chunk, CommandId("native"))
    choice = response.choices[0]
    assert isinstance(choice, StreamingChoiceResponse)
    assert choice.delta.content == "é"
    assert choice.logprobs is not None
    assert choice.logprobs.content == chunk.logprobs_content


async def test_collected_logprobs_keep_every_native_row() -> None:
    chunk = grouped_chunk()

    async def stream() -> AsyncGenerator[TokenChunk]:
        yield chunk

    (payload,) = [
        item async for item in collect_chat_response(CommandId("native"), stream())
    ]
    response = ChatCompletionResponse.model_validate_json(payload)
    choice = response.choices[0]
    assert isinstance(choice, ChatCompletionChoice)
    assert choice.message.content == "é"
    assert choice.logprobs is not None
    assert choice.logprobs.content == chunk.logprobs_content


def test_existing_scalar_logprobs_keep_their_wire_shape() -> None:
    chunk = TokenChunk(
        model=ModelId("test/mlx"), token_id=1, text="hi", usage=None, logprob=-0.3
    )
    choice = chunk_to_response(chunk, CommandId("mlx")).choices[0]
    assert choice.logprobs is not None
    assert choice.logprobs.content == [
        LogprobsContentItem(token="hi", logprob=-0.3, top_logprobs=[])
    ]
