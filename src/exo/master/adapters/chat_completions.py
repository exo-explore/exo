"""OpenAI Chat Completions API adapter for converting requests/responses."""

import time
from collections.abc import AsyncGenerator

from exo.shared.types.api import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionResponse,
    ErrorInfo,
    ErrorResponse,
    FinishReason,
    StreamingChoiceResponse,
)
from exo.shared.types.chunks import TokenChunk
from exo.shared.types.common import CommandId
from exo.shared.types.tasks import ChatCompletionTaskParams


def chat_request_to_internal_params(
    request: ChatCompletionTaskParams,
) -> ChatCompletionTaskParams:
    """Convert Chat Completions API request to internal params.

    This is essentially a pass-through since ChatCompletionTaskParams
    is already the internal format for chat completions.
    """
    return request


def chunk_to_response(
    chunk: TokenChunk, command_id: CommandId
) -> ChatCompletionResponse:
    """Convert a TokenChunk to a streaming ChatCompletionResponse."""
    return ChatCompletionResponse(
        id=command_id,
        created=int(time.time()),
        model=chunk.model,
        choices=[
            StreamingChoiceResponse(
                index=0,
                delta=ChatCompletionMessage(role="assistant", content=chunk.text),
                finish_reason=chunk.finish_reason,
            )
        ],
    )


async def generate_chat_stream(
    command_id: CommandId,
    chunk_stream: AsyncGenerator[TokenChunk, None],
) -> AsyncGenerator[str, None]:
    """Generate Chat Completions API streaming events from TokenChunks."""
    async for chunk in chunk_stream:
        if chunk.finish_reason == "error":
            error_response = ErrorResponse(
                error=ErrorInfo(
                    message=chunk.error_message or "Internal server error",
                    type="InternalServerError",
                    code=500,
                )
            )
            yield f"data: {error_response.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
            return

        chunk_response = chunk_to_response(chunk, command_id)
        yield f"data: {chunk_response.model_dump_json()}\n\n"

        if chunk.finish_reason is not None:
            yield "data: [DONE]\n\n"


async def collect_chat_response(
    command_id: CommandId,
    chunk_stream: AsyncGenerator[TokenChunk, None],
) -> ChatCompletionResponse:
    """Collect all token chunks and return a single ChatCompletionResponse."""
    text_parts: list[str] = []
    model: str | None = None
    finish_reason: FinishReason | None = None
    error_message: str | None = None

    async for chunk in chunk_stream:
        if chunk.finish_reason == "error":
            error_message = chunk.error_message or "Internal server error"
            break

        if model is None:
            model = chunk.model

        text_parts.append(chunk.text)

        if chunk.finish_reason is not None:
            finish_reason = chunk.finish_reason

    if error_message is not None:
        raise ValueError(error_message)

    combined_text = "".join(text_parts)
    assert model is not None

    return ChatCompletionResponse(
        id=command_id,
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=combined_text,
                ),
                finish_reason=finish_reason,
            )
        ],
    )
