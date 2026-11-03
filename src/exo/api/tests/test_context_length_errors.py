# pyright: reportAny=false
"""A request too long for the model's context gets OpenAI's context_length_exceeded error, which
clients and tools recognise: a 400 for a whole response, an error event in a stream (whose
response has started by then), and each API's own error event in the Claude and Responses
streams."""

import json
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from exo.api.adapters.chat_completions import (
    collect_chat_response,
    generate_chat_stream,
)
from exo.api.adapters.claude import generate_claude_stream
from exo.api.adapters.responses import generate_responses_stream
from exo.api.collected_response import CollectedResponse
from exo.shared.types.chunks import (
    ErrorChunk,
    PrefillProgressChunk,
    TokenChunk,
    ToolCallChunk,
)
from exo.shared.types.common import CommandId, ModelId

MODEL = ModelId("test-model")
TOO_LONG = (
    "This model's maximum context length is 131072 tokens. However, your messages "
    "resulted in 160000 tokens. Please reduce the length of the messages."
)
OPENAI_ERROR = {
    "message": TOO_LONG,
    "type": "invalid_request_error",
    "param": "messages",
    "code": "context_length_exceeded",
}

type Chunk = ErrorChunk | TokenChunk | ToolCallChunk | PrefillProgressChunk


def refused() -> ErrorChunk:
    return ErrorChunk(model=MODEL, error_message=TOO_LONG)


def token(text: str, finish: bool = False) -> TokenChunk:
    return TokenChunk(
        model=MODEL,
        text=text,
        token_id=1,
        usage=None,
        finish_reason="stop" if finish else None,
    )


async def chunks(*items: Chunk) -> AsyncGenerator[Chunk, None]:
    for item in items:
        yield item


async def test_a_streamed_request_too_long_for_the_context_ends_with_openais_error() -> (
    None
):
    events = [
        event
        async for event in generate_chat_stream(CommandId("cmd"), chunks(refused()))
    ]

    assert events == [
        f"data: {json.dumps({'error': OPENAI_ERROR}, separators=(',', ':'))}\n\n",
        "data: [DONE]\n\n",
    ]


async def test_a_claude_stream_too_long_for_the_context_ends_with_an_invalid_request() -> (
    None
):
    events = [
        event
        async for event in generate_claude_stream(
            CommandId("cmd"), MODEL, chunks(refused())
        )
    ]

    error = json.loads(events[-1].split("data: ", 1)[1])
    assert error == {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": TOO_LONG,
            "code": "context_length_exceeded",
        },
    }


async def test_a_responses_stream_too_long_for_the_context_fails_with_the_code() -> (
    None
):
    events = [
        event
        async for event in generate_responses_stream(
            CommandId("cmd"), MODEL, chunks(refused())
        )
    ]

    failed = json.loads(events[-1].split("data: ", 1)[1])
    assert failed["type"] == "response.failed"
    assert failed["response"]["error"] == {
        "code": "context_length_exceeded",
        "message": TOO_LONG,
    }


def test_a_whole_response_too_long_for_the_context_gets_a_400() -> None:
    app = FastAPI()
    app.get("/")(
        lambda: CollectedResponse(
            collect_chat_response(CommandId("cmd"), chunks(refused())),
            media_type="application/json",
        )
    )

    reply = TestClient(app).get("/")

    assert reply.status_code == 400
    body: dict[str, Any] = reply.json()
    assert body == {"error": OPENAI_ERROR}
