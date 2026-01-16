"""Tests for OpenAI Responses API conversion functions and types."""

import json
from typing import Any, cast

import pydantic
import pytest

from exo.master.adapters.responses import (
    chat_response_to_responses_response,
    responses_request_to_chat_params,
)
from exo.shared.types.api import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionResponse,
    Usage,
)
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


class TestResponsesRequestToChatParams:
    """Tests for converting OpenAI Responses API requests to ChatCompletionTaskParams."""

    def test_string_input_conversion(self):
        request = ResponsesRequest(
            model="gpt-4o",
            input="Hello, how are you?",
        )
        params = responses_request_to_chat_params(request)

        assert params.model == "gpt-4o"
        assert len(params.messages) == 1
        assert params.messages[0].role == "user"
        assert params.messages[0].content == "Hello, how are you?"

    def test_message_array_input_conversion(self):
        request = ResponsesRequest(
            model="gpt-4o",
            input=[
                ResponseInputMessage(role="user", content="Hello"),
                ResponseInputMessage(role="assistant", content="Hi there!"),
                ResponseInputMessage(role="user", content="How are you?"),
            ],
        )
        params = responses_request_to_chat_params(request)

        assert len(params.messages) == 3
        assert params.messages[0].role == "user"
        assert params.messages[0].content == "Hello"
        assert params.messages[1].role == "assistant"
        assert params.messages[1].content == "Hi there!"
        assert params.messages[2].role == "user"
        assert params.messages[2].content == "How are you?"

    def test_request_with_instructions(self):
        request = ResponsesRequest(
            model="gpt-4o",
            input="Hello",
            instructions="You are a helpful assistant. Be concise.",
        )
        params = responses_request_to_chat_params(request)

        assert len(params.messages) == 2
        assert params.messages[0].role == "system"
        assert params.messages[0].content == "You are a helpful assistant. Be concise."
        assert params.messages[1].role == "user"
        assert params.messages[1].content == "Hello"

    def test_request_with_optional_parameters(self):
        request = ResponsesRequest(
            model="gpt-4o",
            input="Hello",
            max_output_tokens=500,
            temperature=0.8,
            top_p=0.95,
            stream=True,
        )
        params = responses_request_to_chat_params(request)

        assert params.max_tokens == 500
        assert params.temperature == 0.8
        assert params.top_p == 0.95
        assert params.stream is True

    def test_request_with_system_role_in_messages(self):
        request = ResponsesRequest(
            model="gpt-4o",
            input=[
                ResponseInputMessage(role="system", content="Be helpful"),
                ResponseInputMessage(role="user", content="Hello"),
            ],
        )
        params = responses_request_to_chat_params(request)

        assert len(params.messages) == 2
        assert params.messages[0].role == "system"
        assert params.messages[1].role == "user"

    def test_request_with_developer_role(self):
        request = ResponsesRequest(
            model="gpt-4o",
            input=[
                ResponseInputMessage(role="developer", content="Internal note"),
                ResponseInputMessage(role="user", content="Hello"),
            ],
        )
        params = responses_request_to_chat_params(request)

        assert len(params.messages) == 2
        assert params.messages[0].role == "developer"


class TestChatResponseToResponsesResponse:
    """Tests for converting ChatCompletionResponse to OpenAI Responses API response."""

    def test_basic_response_conversion(self):
        response = ChatCompletionResponse(
            id="chatcmpl-123",
            created=1234567890,
            model="llama-3.2-1b",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content="Hello! How can I help you?",
                    ),
                    finish_reason="stop",
                )
            ],
        )
        responses_response = chat_response_to_responses_response(response)

        assert responses_response.id == "resp_chatcmpl-123"
        assert responses_response.object == "response"
        assert responses_response.model == "llama-3.2-1b"
        assert responses_response.status == "completed"
        assert responses_response.output_text == "Hello! How can I help you?"
        assert len(responses_response.output) == 1
        assert responses_response.output[0].type == "message"
        assert responses_response.output[0].role == "assistant"
        assert len(responses_response.output[0].content) == 1
        assert responses_response.output[0].content[0].type == "output_text"
        assert (
            responses_response.output[0].content[0].text == "Hello! How can I help you?"
        )

    def test_response_with_usage(self):
        response = ChatCompletionResponse(
            id="chatcmpl-123",
            created=1234567890,
            model="llama-3.2-1b",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="Hello!"),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
        )
        responses_response = chat_response_to_responses_response(response)

        assert responses_response.usage is not None
        assert responses_response.usage.input_tokens == 10
        assert responses_response.usage.output_tokens == 5
        assert responses_response.usage.total_tokens == 15

    def test_response_with_empty_content(self):
        response = ChatCompletionResponse(
            id="chatcmpl-123",
            created=1234567890,
            model="llama-3.2-1b",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content=""),
                    finish_reason="stop",
                )
            ],
        )
        responses_response = chat_response_to_responses_response(response)

        assert responses_response.output_text == ""
        assert responses_response.output[0].content[0].text == ""

    def test_response_with_no_choices(self):
        response = ChatCompletionResponse(
            id="chatcmpl-123",
            created=1234567890,
            model="llama-3.2-1b",
            choices=[],
        )
        responses_response = chat_response_to_responses_response(response)

        assert responses_response.output_text == ""

    def test_response_without_usage(self):
        response = ChatCompletionResponse(
            id="chatcmpl-123",
            created=1234567890,
            model="llama-3.2-1b",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="Hello!"),
                    finish_reason="stop",
                )
            ],
        )
        responses_response = chat_response_to_responses_response(response)

        assert responses_response.usage is None

    def test_response_item_id_format(self):
        response = ChatCompletionResponse(
            id="chatcmpl-abc123",
            created=1234567890,
            model="llama-3.2-1b",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="Hello!"),
                    finish_reason="stop",
                )
            ],
        )
        responses_response = chat_response_to_responses_response(response)

        assert responses_response.output[0].id == "item_chatcmpl-abc123"


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
            model="gpt-4o",
            input="Hello",
        )
        assert request.input == "Hello"

    def test_request_accepts_message_array_input(self):
        request = ResponsesRequest(
            model="gpt-4o",
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
