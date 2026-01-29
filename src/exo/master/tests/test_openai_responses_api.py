"""Tests for OpenAI Responses API types.

ResponsesRequest is the canonical internal type used throughout the pipeline.
No conversion is needed for Responses API requests.
"""

import json
from typing import Any, cast

import pydantic
import pytest

from exo.shared.types.common import ModelId
from exo.shared.types.openai_responses import (
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseCreatedEvent,
    ResponseInputMessage,
    ResponseMessageItem,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputText,
    ResponsesRequest,
    ResponsesResponse,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseUsage,
)


class TestResponsesRequestAsCanonicalType:
    """Tests for ResponsesRequest as the canonical internal type."""

    def test_string_input(self):
        request = ResponsesRequest(
            model=ModelId("gpt-4o"),
            input="Hello, how are you?",
        )

        assert request.model == "gpt-4o"
        assert request.input == "Hello, how are you?"
        assert request.instructions is None

    def test_message_array_input(self):
        request = ResponsesRequest(
            model=ModelId("gpt-4o"),
            input=[
                ResponseInputMessage(role="user", content="Hello"),
                ResponseInputMessage(role="assistant", content="Hi there!"),
                ResponseInputMessage(role="user", content="How are you?"),
            ],
        )

        assert isinstance(request.input, list)
        assert len(request.input) == 3
        assert request.input[0].role == "user"
        assert request.input[0].content == "Hello"
        assert request.input[1].role == "assistant"
        assert request.input[1].content == "Hi there!"
        assert request.input[2].role == "user"
        assert request.input[2].content == "How are you?"

    def test_request_with_instructions(self):
        request = ResponsesRequest(
            model=ModelId("gpt-4o"),
            input="Hello",
            instructions="You are a helpful assistant. Be concise.",
        )

        assert request.input == "Hello"
        assert request.instructions == "You are a helpful assistant. Be concise."

    def test_request_with_optional_parameters(self):
        request = ResponsesRequest(
            model=ModelId("gpt-4o"),
            input="Hello",
            max_output_tokens=500,
            temperature=0.8,
            top_p=0.95,
            stream=True,
        )

        assert request.max_output_tokens == 500
        assert request.temperature == 0.8
        assert request.top_p == 0.95
        assert request.stream is True

    def test_request_with_new_fields(self):
        """Test the additional fields added for internal use."""
        request = ResponsesRequest(
            model=ModelId("gpt-4o"),
            input="Hello",
            top_k=40,
            seed=42,
            stop=["STOP", "END"],
            tools=[{"type": "function", "function": {"name": "test"}}],
        )

        assert request.top_k == 40
        assert request.seed == 42
        assert request.stop == ["STOP", "END"]
        assert request.tools == [{"type": "function", "function": {"name": "test"}}]

    def test_request_with_system_role_in_messages(self):
        request = ResponsesRequest(
            model=ModelId("gpt-4o"),
            input=[
                ResponseInputMessage(role="system", content="Be helpful"),
                ResponseInputMessage(role="user", content="Hello"),
            ],
        )

        assert isinstance(request.input, list)
        assert len(request.input) == 2
        assert request.input[0].role == "system"
        assert request.input[1].role == "user"

    def test_request_with_developer_role(self):
        request = ResponsesRequest(
            model=ModelId("gpt-4o"),
            input=[
                ResponseInputMessage(role="developer", content="Internal note"),
                ResponseInputMessage(role="user", content="Hello"),
            ],
        )

        assert isinstance(request.input, list)
        assert len(request.input) == 2
        assert request.input[0].role == "developer"


class TestResponsesRequestValidation:
    """Tests for OpenAI Responses API request validation."""

    def test_request_requires_model(self):
        with pytest.raises(pydantic.ValidationError):
            ResponsesRequest.model_validate(
                {
                    "input": "Hello",
                }
            )

    def test_request_requires_input(self):
        with pytest.raises(pydantic.ValidationError):
            ResponsesRequest.model_validate(
                {
                    "model": "gpt-4o",
                }
            )

    def test_request_accepts_string_input(self):
        request = ResponsesRequest(
            model=ModelId("gpt-4o"),
            input="Hello",
        )
        assert request.input == "Hello"

    def test_request_accepts_message_array_input(self):
        request = ResponsesRequest(
            model=ModelId("gpt-4o"),
            input=[ResponseInputMessage(role="user", content="Hello")],
        )
        assert len(request.input) == 1


class TestResponsesStreamingEvents:
    """Tests for OpenAI Responses API streaming event serialization."""

    def test_response_created_event_format(self):
        response = ResponsesResponse(
            id="resp_123",
            model="gpt-4o",
            status="in_progress",
            output=[],
            output_text="",
        )
        event = ResponseCreatedEvent(response=response)
        json_str = event.model_dump_json()
        parsed = cast(dict[str, Any], json.loads(json_str))

        assert parsed["type"] == "response.created"
        assert parsed["response"]["id"] == "resp_123"
        assert parsed["response"]["object"] == "response"
        assert parsed["response"]["status"] == "in_progress"

    def test_output_item_added_event_format(self):
        item = ResponseMessageItem(
            id="item_123",
            content=[ResponseOutputText(text="")],
            status="in_progress",
        )
        event = ResponseOutputItemAddedEvent(output_index=0, item=item)
        json_str = event.model_dump_json()
        parsed = cast(dict[str, Any], json.loads(json_str))

        assert parsed["type"] == "response.output_item.added"
        assert parsed["output_index"] == 0
        assert parsed["item"]["type"] == "message"
        assert parsed["item"]["id"] == "item_123"
        assert parsed["item"]["role"] == "assistant"

    def test_content_part_added_event_format(self):
        part = ResponseOutputText(text="")
        event = ResponseContentPartAddedEvent(
            output_index=0,
            content_index=0,
            part=part,
        )
        json_str = event.model_dump_json()
        parsed = cast(dict[str, Any], json.loads(json_str))

        assert parsed["type"] == "response.content_part.added"
        assert parsed["output_index"] == 0
        assert parsed["content_index"] == 0
        assert parsed["part"]["type"] == "output_text"

    def test_text_delta_event_format(self):
        event = ResponseTextDeltaEvent(
            output_index=0,
            content_index=0,
            delta="Hello",
        )
        json_str = event.model_dump_json()
        parsed = cast(dict[str, Any], json.loads(json_str))

        assert parsed["type"] == "response.output_text.delta"
        assert parsed["output_index"] == 0
        assert parsed["content_index"] == 0
        assert parsed["delta"] == "Hello"

    def test_text_done_event_format(self):
        event = ResponseTextDoneEvent(
            output_index=0,
            content_index=0,
            text="Hello, world!",
        )
        json_str = event.model_dump_json()
        parsed = cast(dict[str, Any], json.loads(json_str))

        assert parsed["type"] == "response.output_text.done"
        assert parsed["text"] == "Hello, world!"

    def test_output_item_done_event_format(self):
        item = ResponseMessageItem(
            id="item_123",
            content=[ResponseOutputText(text="Hello, world!")],
            status="completed",
        )
        event = ResponseOutputItemDoneEvent(output_index=0, item=item)
        json_str = event.model_dump_json()
        parsed = cast(dict[str, Any], json.loads(json_str))

        assert parsed["type"] == "response.output_item.done"
        assert parsed["item"]["status"] == "completed"
        assert parsed["item"]["content"][0]["text"] == "Hello, world!"

    def test_response_completed_event_format(self):
        item = ResponseMessageItem(
            id="item_123",
            content=[ResponseOutputText(text="Hello!")],
            status="completed",
        )
        response = ResponsesResponse(
            id="resp_123",
            model="gpt-4o",
            status="completed",
            output=[item],
            output_text="Hello!",
            usage=ResponseUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        event = ResponseCompletedEvent(response=response)
        json_str = event.model_dump_json()
        parsed = cast(dict[str, Any], json.loads(json_str))

        assert parsed["type"] == "response.completed"
        assert parsed["response"]["status"] == "completed"
        assert parsed["response"]["output_text"] == "Hello!"
        assert parsed["response"]["usage"]["total_tokens"] == 15

    def test_sse_format(self):
        """Test that SSE format is correctly generated."""
        event = ResponseTextDeltaEvent(
            output_index=0,
            content_index=0,
            delta="Hello",
        )
        # Simulate the SSE format used in the streaming generator
        sse_line = (
            f"event: response.output_text.delta\ndata: {event.model_dump_json()}\n\n"
        )

        assert sse_line.startswith("event: response.output_text.delta\n")
        assert "data: " in sse_line
        assert sse_line.endswith("\n\n")
