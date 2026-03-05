"""Tests for Claude Messages API tool_use support in the adapter."""

import json
from collections.abc import AsyncGenerator
from typing import Any, cast

from exo.master.adapters.claude import (
    ClaudeMessagesResponse,
    collect_claude_response,
    generate_claude_stream,
)
from exo.shared.types.api import ToolCallItem
from exo.shared.types.chunks import ErrorChunk, TokenChunk, ToolCallChunk
from exo.shared.types.common import CommandId, ModelId


async def _chunks_to_stream(
    chunks: list[ErrorChunk | ToolCallChunk | TokenChunk],
) -> AsyncGenerator[ErrorChunk | ToolCallChunk | TokenChunk, None]:
    for chunk in chunks:
        yield chunk


async def _collect_response(
    command_id: CommandId,
    model: str,
    chunk_stream: AsyncGenerator[ErrorChunk | ToolCallChunk | TokenChunk, None],
) -> ClaudeMessagesResponse:
    """Helper to consume the async generator and parse the JSON response."""
    parts: list[str] = []
    async for part in collect_claude_response(command_id, model, chunk_stream):
        parts.append(part)
    return ClaudeMessagesResponse.model_validate_json("".join(parts))


MODEL = ModelId("test-model")
COMMAND_ID = CommandId("cmd_test123")


def _parse_sse_events(events: list[str]) -> list[dict[str, Any]]:
    """Parse SSE event strings into JSON dicts."""
    parsed: list[dict[str, Any]] = []
    for event_str in events:
        for line in event_str.strip().split("\n"):
            if line.startswith("data: "):
                parsed.append(cast(dict[str, Any], json.loads(line[6:])))
    return parsed


class TestCollectClaudeResponseToolUse:
    """Tests for non-streaming tool_use response collection."""

    async def test_tool_call_chunk_produces_tool_use_blocks(self):
        chunks: list[ErrorChunk | ToolCallChunk | TokenChunk] = [
            ToolCallChunk(
                model=MODEL,
                usage=None,
                tool_calls=[
                    ToolCallItem(
                        name="get_weather",
                        arguments='{"location": "San Francisco"}',
                    )
                ],
            ),
        ]
        response = await _collect_response(
            COMMAND_ID, "test-model", _chunks_to_stream(chunks)
        )

        assert response.stop_reason == "tool_use"
        tool_blocks = [b for b in response.content if b.type == "tool_use"]
        assert len(tool_blocks) == 1
        block = tool_blocks[0]
        assert block.type == "tool_use"
        assert block.name == "get_weather"
        assert block.input == {"location": "San Francisco"}
        assert block.id.startswith("toolu_")

    async def test_multiple_tool_calls(self):
        chunks: list[ErrorChunk | ToolCallChunk | TokenChunk] = [
            ToolCallChunk(
                model=MODEL,
                usage=None,
                tool_calls=[
                    ToolCallItem(
                        name="get_weather",
                        arguments='{"location": "SF"}',
                    ),
                    ToolCallItem(
                        name="get_time",
                        arguments='{"timezone": "PST"}',
                    ),
                ],
            ),
        ]
        response = await _collect_response(
            COMMAND_ID, "test-model", _chunks_to_stream(chunks)
        )

        assert response.stop_reason == "tool_use"
        tool_blocks = [b for b in response.content if b.type == "tool_use"]
        assert len(tool_blocks) == 2
        assert tool_blocks[0].name == "get_weather"
        assert tool_blocks[1].name == "get_time"

    async def test_mixed_text_and_tool_use(self):
        chunks: list[ErrorChunk | ToolCallChunk | TokenChunk] = [
            TokenChunk(model=MODEL, text="Let me check ", token_id=1, usage=None),
            TokenChunk(model=MODEL, text="the weather.", token_id=2, usage=None),
            ToolCallChunk(
                model=MODEL,
                usage=None,
                tool_calls=[
                    ToolCallItem(
                        name="get_weather",
                        arguments='{"location": "NYC"}',
                    )
                ],
            ),
        ]
        response = await _collect_response(
            COMMAND_ID, "test-model", _chunks_to_stream(chunks)
        )

        assert response.stop_reason == "tool_use"
        text_blocks = [b for b in response.content if b.type == "text"]
        tool_blocks = [b for b in response.content if b.type == "tool_use"]
        assert len(text_blocks) == 1
        assert text_blocks[0].text == "Let me check the weather."
        assert len(tool_blocks) == 1
        assert tool_blocks[0].name == "get_weather"

    async def test_no_content_produces_empty_text_block(self):
        chunks: list[ErrorChunk | ToolCallChunk | TokenChunk] = []
        response = await _collect_response(
            COMMAND_ID, "test-model", _chunks_to_stream(chunks)
        )
        assert len(response.content) == 1
        assert response.content[0].type == "text"


class TestGenerateClaudeStreamToolUse:
    """Tests for streaming tool_use event generation."""

    async def test_tool_call_emits_tool_use_events(self):
        chunks: list[ErrorChunk | ToolCallChunk | TokenChunk] = [
            ToolCallChunk(
                model=MODEL,
                usage=None,
                tool_calls=[
                    ToolCallItem(
                        name="get_weather",
                        arguments='{"location": "SF"}',
                    )
                ],
            ),
        ]
        events: list[str] = []
        async for event in generate_claude_stream(
            COMMAND_ID, "test-model", _chunks_to_stream(chunks)
        ):
            events.append(event)

        parsed = _parse_sse_events(events)

        # Find tool_use content_block_start
        tool_starts = [
            e
            for e in parsed
            if e.get("type") == "content_block_start"
            and cast(dict[str, Any], e.get("content_block", {})).get("type")
            == "tool_use"
        ]
        assert len(tool_starts) == 1
        content_block = cast(dict[str, Any], tool_starts[0]["content_block"])
        assert content_block["name"] == "get_weather"
        assert content_block["input"] == {}
        assert cast(str, content_block["id"]).startswith("toolu_")

        # Find input_json_delta
        json_deltas = [
            e
            for e in parsed
            if e.get("type") == "content_block_delta"
            and cast(dict[str, Any], e.get("delta", {})).get("type")
            == "input_json_delta"
        ]
        assert len(json_deltas) == 1
        delta = cast(dict[str, Any], json_deltas[0]["delta"])
        assert json.loads(cast(str, delta["partial_json"])) == {"location": "SF"}

        # Find message_delta with tool_use stop reason
        msg_deltas = [e for e in parsed if e.get("type") == "message_delta"]
        assert len(msg_deltas) == 1
        assert cast(dict[str, Any], msg_deltas[0]["delta"])["stop_reason"] == "tool_use"

    async def test_streaming_mixed_text_and_tool_use(self):
        chunks: list[ErrorChunk | ToolCallChunk | TokenChunk] = [
            TokenChunk(model=MODEL, text="Hello ", token_id=1, usage=None),
            ToolCallChunk(
                model=MODEL,
                usage=None,
                tool_calls=[
                    ToolCallItem(
                        name="search",
                        arguments='{"query": "test"}',
                    )
                ],
            ),
        ]
        events: list[str] = []
        async for event in generate_claude_stream(
            COMMAND_ID, "test-model", _chunks_to_stream(chunks)
        ):
            events.append(event)

        parsed = _parse_sse_events(events)

        # Should have text delta at index 0
        text_deltas = [
            e
            for e in parsed
            if e.get("type") == "content_block_delta"
            and cast(dict[str, Any], e.get("delta", {})).get("type") == "text_delta"
        ]
        assert len(text_deltas) == 1
        assert text_deltas[0]["index"] == 0
        assert cast(dict[str, Any], text_deltas[0]["delta"])["text"] == "Hello "

        # Tool block at index 1
        tool_starts = [
            e
            for e in parsed
            if e.get("type") == "content_block_start"
            and cast(dict[str, Any], e.get("content_block", {})).get("type")
            == "tool_use"
        ]
        assert len(tool_starts) == 1
        assert tool_starts[0]["index"] == 1

        # Stop reason should be tool_use
        msg_deltas = [e for e in parsed if e.get("type") == "message_delta"]
        assert cast(dict[str, Any], msg_deltas[0]["delta"])["stop_reason"] == "tool_use"

    async def test_streaming_tool_block_stop_events(self):
        chunks: list[ErrorChunk | ToolCallChunk | TokenChunk] = [
            ToolCallChunk(
                model=MODEL,
                usage=None,
                tool_calls=[
                    ToolCallItem(name="fn1", arguments="{}"),
                    ToolCallItem(name="fn2", arguments='{"a": 1}'),
                ],
            ),
        ]
        events: list[str] = []
        async for event in generate_claude_stream(
            COMMAND_ID, "test-model", _chunks_to_stream(chunks)
        ):
            events.append(event)

        parsed = _parse_sse_events(events)

        # Two tool block starts (at indices 0 and 1 — no text block when only tools)
        tool_starts = [
            e
            for e in parsed
            if e.get("type") == "content_block_start"
            and cast(dict[str, Any], e.get("content_block", {})).get("type")
            == "tool_use"
        ]
        assert len(tool_starts) == 2
        assert tool_starts[0]["index"] == 0
        assert tool_starts[1]["index"] == 1

        # Two tool block stops (at indices 0 and 1)
        block_stops = [e for e in parsed if e.get("type") == "content_block_stop"]
        stop_indices = [e["index"] for e in block_stops]
        assert 0 in stop_indices
        assert 1 in stop_indices
