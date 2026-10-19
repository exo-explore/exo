"""Tests for stop-sequence reporting in the Claude Messages API adapter.

When generation stops because a user-supplied stop sequence matched, the
response must report stop_reason="stop_sequence" and echo the matched sequence
in the stop_sequence field — not the generic "end_turn" used for a natural EOS.
"""

import json
from collections.abc import AsyncGenerator
from typing import Any, cast

from exo.api.adapters.claude import (
    ClaudeMessagesResponse,
    collect_claude_response,
    generate_claude_stream,
)
from exo.shared.types.chunks import ErrorChunk, TokenChunk, ToolCallChunk
from exo.shared.types.common import CommandId, ModelId

MODEL = ModelId("test-model")
COMMAND_ID = CommandId("cmd_test123")


async def _chunks_to_stream(
    chunks: list[ErrorChunk | ToolCallChunk | TokenChunk],
) -> AsyncGenerator[ErrorChunk | ToolCallChunk | TokenChunk, None]:
    for chunk in chunks:
        yield chunk


async def _collect_response(
    chunks: list[ErrorChunk | ToolCallChunk | TokenChunk],
) -> ClaudeMessagesResponse:
    parts: list[str] = []
    async for part in collect_claude_response(
        COMMAND_ID, MODEL, _chunks_to_stream(chunks)
    ):
        parts.append(part)
    return ClaudeMessagesResponse.model_validate_json("".join(parts))


def _parse_sse_events(events: list[str]) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for event_str in events:
        for line in event_str.strip().split("\n"):
            if line.startswith("data: "):
                parsed.append(cast(dict[str, Any], json.loads(line[6:])))
    return parsed


class TestCollectClaudeResponseStopSequence:
    async def test_matched_stop_sequence_reports_stop_sequence(self):
        chunks: list[ErrorChunk | ToolCallChunk | TokenChunk] = [
            TokenChunk(model=MODEL, text="ABC", token_id=1, usage=None),
            TokenChunk(
                model=MODEL,
                text="",
                token_id=2,
                usage=None,
                finish_reason="stop",
                matched_stop_sequence="END",
            ),
        ]
        response = await _collect_response(chunks)

        assert response.stop_reason == "stop_sequence"
        assert response.stop_sequence == "END"
        # The stop sequence itself is never part of the emitted text.
        text_blocks = [b for b in response.content if b.type == "text"]
        assert "END" not in "".join(b.text for b in text_blocks)

    async def test_natural_eos_still_reports_end_turn(self):
        chunks: list[ErrorChunk | ToolCallChunk | TokenChunk] = [
            TokenChunk(model=MODEL, text="Hello", token_id=1, usage=None),
            TokenChunk(
                model=MODEL, text="", token_id=2, usage=None, finish_reason="stop"
            ),
        ]
        response = await _collect_response(chunks)

        assert response.stop_reason == "end_turn"
        assert response.stop_sequence is None

    async def test_length_limit_reports_max_tokens(self):
        chunks: list[ErrorChunk | ToolCallChunk | TokenChunk] = [
            TokenChunk(
                model=MODEL,
                text="Hello",
                token_id=1,
                usage=None,
                finish_reason="length",
            ),
        ]
        response = await _collect_response(chunks)

        assert response.stop_reason == "max_tokens"
        assert response.stop_sequence is None


class TestStreamingClaudeResponseStopSequence:
    async def _stream_events(
        self, chunks: list[ErrorChunk | ToolCallChunk | TokenChunk]
    ) -> list[dict[str, Any]]:
        events: list[str] = []
        async for event in generate_claude_stream(
            COMMAND_ID, MODEL, _chunks_to_stream(chunks)
        ):
            events.append(event)
        return _parse_sse_events(events)

    async def test_streaming_matched_stop_sequence(self):
        chunks: list[ErrorChunk | ToolCallChunk | TokenChunk] = [
            TokenChunk(model=MODEL, text="ABC", token_id=1, usage=None),
            TokenChunk(
                model=MODEL,
                text="",
                token_id=2,
                usage=None,
                finish_reason="stop",
                matched_stop_sequence="END",
            ),
        ]
        parsed = await self._stream_events(chunks)

        message_deltas = [p for p in parsed if p.get("type") == "message_delta"]
        assert len(message_deltas) == 1
        delta = cast(dict[str, Any], message_deltas[0]["delta"])
        assert delta["stop_reason"] == "stop_sequence"
        assert delta["stop_sequence"] == "END"

    async def test_streaming_natural_eos(self):
        chunks: list[ErrorChunk | ToolCallChunk | TokenChunk] = [
            TokenChunk(model=MODEL, text="Hi", token_id=1, usage=None),
            TokenChunk(
                model=MODEL, text="", token_id=2, usage=None, finish_reason="stop"
            ),
        ]
        parsed = await self._stream_events(chunks)

        message_deltas = [p for p in parsed if p.get("type") == "message_delta"]
        delta = cast(dict[str, Any], message_deltas[0]["delta"])
        assert delta["stop_reason"] == "end_turn"
        assert delta["stop_sequence"] is None
