from collections.abc import AsyncGenerator

from exo.api.adapters.chat_completions import collect_chat_response
from exo.api.types import ErrorResponse
from exo.shared.types.chunks import (
    ErrorChunk,
    PrefillProgressChunk,
    TokenChunk,
    ToolCallChunk,
)
from exo.shared.types.common import CommandId, ModelId


async def _stream(
    *chunks: ErrorChunk | ToolCallChunk | TokenChunk | PrefillProgressChunk,
) -> AsyncGenerator[
    ErrorChunk | ToolCallChunk | TokenChunk | PrefillProgressChunk, None
]:
    for chunk in chunks:
        yield chunk


async def test_error_chunk_yields_error_body_instead_of_raising() -> None:
    """A request rejected before the first token must still produce a body.

    The non-streaming endpoint wraps this generator in a StreamingResponse, so
    an exception here would leave the client with an empty HTTP 200.
    """
    error = ErrorChunk(
        model=ModelId("test/model"),
        finish_reason="error",
        error_message="image input is not supported here",
    )
    items = [
        item async for item in collect_chat_response(CommandId("cmd"), _stream(error))
    ]
    assert len(items) == 1
    body = ErrorResponse.model_validate_json(items[0])
    assert body.error.message == "image input is not supported here"
    assert body.error.code == 500
