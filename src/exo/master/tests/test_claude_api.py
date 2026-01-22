"""Tests for Claude Messages API conversion functions and types."""

import json
from typing import Any, cast

import pydantic
import pytest

from exo.master.adapters.claude import (
    claude_request_to_internal,
    finish_reason_to_claude_stop_reason,
)
from exo.shared.types.claude_api import (
    ClaudeContentBlockDeltaEvent,
    ClaudeContentBlockStartEvent,
    ClaudeContentBlockStopEvent,
    ClaudeMessage,
    ClaudeMessageDelta,
    ClaudeMessageDeltaEvent,
    ClaudeMessageDeltaUsage,
    ClaudeMessagesRequest,
    ClaudeMessageStart,
    ClaudeMessageStartEvent,
    ClaudeMessageStopEvent,
    ClaudeTextBlock,
    ClaudeTextDelta,
    ClaudeUsage,
)
from exo.shared.types.common import ModelId


class TestFinishReasonToClaudeStopReason:
    """Tests for finish_reason to Claude stop_reason mapping."""

    def test_stop_maps_to_end_turn(self):
        assert finish_reason_to_claude_stop_reason("stop") == "end_turn"

    def test_length_maps_to_max_tokens(self):
        assert finish_reason_to_claude_stop_reason("length") == "max_tokens"

    def test_tool_calls_maps_to_tool_use(self):
        assert finish_reason_to_claude_stop_reason("tool_calls") == "tool_use"

    def test_function_call_maps_to_tool_use(self):
        assert finish_reason_to_claude_stop_reason("function_call") == "tool_use"

    def test_content_filter_maps_to_end_turn(self):
        assert finish_reason_to_claude_stop_reason("content_filter") == "end_turn"

    def test_none_returns_none(self):
        assert finish_reason_to_claude_stop_reason(None) is None


class TestClaudeRequestToInternal:
    """Tests for converting Claude Messages API requests to ResponsesRequest."""

    def test_basic_request_conversion(self):
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            messages=[
                ClaudeMessage(role="user", content="Hello"),
            ],
        )
        params = claude_request_to_internal(request)

        assert params.model == "claude-3-opus"
        assert params.max_output_tokens == 100
        assert isinstance(params.input, list)
        assert len(params.input) == 1
        assert params.input[0].role == "user"
        assert params.input[0].content == "Hello"
        assert params.instructions is None

    def test_request_with_system_string(self):
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            system="You are a helpful assistant.",
            messages=[
                ClaudeMessage(role="user", content="Hello"),
            ],
        )
        params = claude_request_to_internal(request)

        assert params.instructions == "You are a helpful assistant."
        assert isinstance(params.input, list)
        assert len(params.input) == 1
        assert params.input[0].role == "user"
        assert params.input[0].content == "Hello"

    def test_request_with_system_text_blocks(self):
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            system=[
                ClaudeTextBlock(text="You are helpful. "),
                ClaudeTextBlock(text="Be concise."),
            ],
            messages=[
                ClaudeMessage(role="user", content="Hello"),
            ],
        )
        params = claude_request_to_internal(request)

        assert params.instructions == "You are helpful. Be concise."
        assert isinstance(params.input, list)
        assert len(params.input) == 1

    def test_request_with_content_blocks(self):
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            messages=[
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeTextBlock(text="First part. "),
                        ClaudeTextBlock(text="Second part."),
                    ],
                ),
            ],
        )
        params = claude_request_to_internal(request)

        assert isinstance(params.input, list)
        assert len(params.input) == 1
        assert params.input[0].content == "First part. Second part."

    def test_request_with_multi_turn_conversation(self):
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            messages=[
                ClaudeMessage(role="user", content="Hello"),
                ClaudeMessage(role="assistant", content="Hi there!"),
                ClaudeMessage(role="user", content="How are you?"),
            ],
        )
        params = claude_request_to_internal(request)

        assert isinstance(params.input, list)
        assert len(params.input) == 3
        assert params.input[0].role == "user"
        assert params.input[1].role == "assistant"
        assert params.input[2].role == "user"

    def test_request_with_optional_parameters(self):
        request = ClaudeMessagesRequest(
            model=ModelId("claude-3-opus"),
            max_tokens=100,
            messages=[ClaudeMessage(role="user", content="Hello")],
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            stop_sequences=["STOP", "END"],
            stream=True,
        )
        params = claude_request_to_internal(request)

        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.top_k == 40
        assert params.stop == ["STOP", "END"]
        assert params.stream is True


class TestClaudeMessagesRequestValidation:
    """Tests for Claude Messages API request validation."""

    def test_request_requires_model(self):
        with pytest.raises(pydantic.ValidationError):
            ClaudeMessagesRequest.model_validate(
                {
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hello"}],
                }
            )

    def test_request_requires_max_tokens(self):
        with pytest.raises(pydantic.ValidationError):
            ClaudeMessagesRequest.model_validate(
                {
                    "model": "claude-3-opus",
                    "messages": [{"role": "user", "content": "Hello"}],
                }
            )

    def test_request_requires_messages(self):
        with pytest.raises(pydantic.ValidationError):
            ClaudeMessagesRequest.model_validate(
                {
                    "model": "claude-3-opus",
                    "max_tokens": 100,
                }
            )


class TestClaudeStreamingEvents:
    """Tests for Claude Messages API streaming event serialization."""

    def test_message_start_event_format(self):
        message = ClaudeMessageStart(
            id="msg_123",
            model="claude-3-opus",
            content=[],
            stop_reason=None,
            usage=ClaudeUsage(input_tokens=10, output_tokens=0),
        )
        event = ClaudeMessageStartEvent(message=message)
        json_str = event.model_dump_json()
        parsed = cast(dict[str, Any], json.loads(json_str))

        assert parsed["type"] == "message_start"
        assert parsed["message"]["id"] == "msg_123"
        assert parsed["message"]["type"] == "message"
        assert parsed["message"]["role"] == "assistant"
        assert parsed["message"]["model"] == "claude-3-opus"

    def test_content_block_start_event_format(self):
        event = ClaudeContentBlockStartEvent(
            index=0,
            content_block=ClaudeTextBlock(text=""),
        )
        json_str = event.model_dump_json()
        parsed = cast(dict[str, Any], json.loads(json_str))

        assert parsed["type"] == "content_block_start"
        assert parsed["index"] == 0
        assert parsed["content_block"]["type"] == "text"
        assert parsed["content_block"]["text"] == ""

    def test_content_block_delta_event_format(self):
        event = ClaudeContentBlockDeltaEvent(
            index=0,
            delta=ClaudeTextDelta(text="Hello"),
        )
        json_str = event.model_dump_json()
        parsed = cast(dict[str, Any], json.loads(json_str))

        assert parsed["type"] == "content_block_delta"
        assert parsed["index"] == 0
        assert parsed["delta"]["type"] == "text_delta"
        assert parsed["delta"]["text"] == "Hello"

    def test_content_block_stop_event_format(self):
        event = ClaudeContentBlockStopEvent(index=0)
        json_str = event.model_dump_json()
        parsed = cast(dict[str, Any], json.loads(json_str))

        assert parsed["type"] == "content_block_stop"
        assert parsed["index"] == 0

    def test_message_delta_event_format(self):
        event = ClaudeMessageDeltaEvent(
            delta=ClaudeMessageDelta(stop_reason="end_turn"),
            usage=ClaudeMessageDeltaUsage(output_tokens=25),
        )
        json_str = event.model_dump_json()
        parsed = cast(dict[str, Any], json.loads(json_str))

        assert parsed["type"] == "message_delta"
        assert parsed["delta"]["stop_reason"] == "end_turn"
        assert parsed["usage"]["output_tokens"] == 25

    def test_message_stop_event_format(self):
        event = ClaudeMessageStopEvent()
        json_str = event.model_dump_json()
        parsed = cast(dict[str, Any], json.loads(json_str))

        assert parsed["type"] == "message_stop"

    def test_sse_format(self):
        """Test that SSE format is correctly generated."""
        event = ClaudeContentBlockDeltaEvent(
            index=0,
            delta=ClaudeTextDelta(text="Hello"),
        )
        # Simulate the SSE format used in the streaming generator
        sse_line = f"event: content_block_delta\ndata: {event.model_dump_json()}\n\n"

        assert sse_line.startswith("event: content_block_delta\n")
        assert "data: " in sse_line
        assert sse_line.endswith("\n\n")
