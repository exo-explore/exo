"""review2: a refused (context_length_exceeded) or failed generation, streamed through the
Claude Messages or OpenAI Responses endpoint, ends as a normal, empty, successful reply: the
adapters `break` on an ErrorChunk and then emit message_stop / response.completed. The
context-length fix makes this reachable for requests that used to run, and the error-code fix
doesn't reach these endpoints."""

from collections.abc import AsyncGenerator

from exo.api.adapters.claude import generate_claude_stream
from exo.api.adapters.responses import generate_responses_stream
from exo.shared.types.chunks import (
    ErrorChunk,
    PrefillProgressChunk,
    TokenChunk,
    ToolCallChunk,
)
from exo.shared.types.common import CommandId, ModelId

MODEL = ModelId("test-model")
REFUSAL = ErrorChunk(
    model=MODEL,
    error_message="This model's maximum context length is 32768 tokens. However, your messages resulted in 40000 tokens. Please reduce the length of the messages.",
)


async def _refused() -> AsyncGenerator[
    ErrorChunk | ToolCallChunk | TokenChunk | PrefillProgressChunk, None
]:
    yield REFUSAL


async def test_claude_stream_reports_a_refused_request_as_an_error() -> None:
    events = [
        e async for e in generate_claude_stream(CommandId("c"), MODEL, _refused())
    ]
    text = "".join(events)
    assert "event: error" in text, f"no error event; stream was:\n{text}"
    assert "event: message_stop" not in text


async def test_responses_stream_reports_a_refused_request_as_an_error() -> None:
    events = [
        e async for e in generate_responses_stream(CommandId("c"), MODEL, _refused())
    ]
    text = "".join(events)
    assert "context length" in text or "error" in text.split("response.completed")[0], (
        f"no error in the stream:\n{text}"
    )
    assert "response.completed" not in text
