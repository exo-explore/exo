"""Tests for OpenAI Responses API wire types.

ResponsesRequest is the API wire type for the Responses endpoint.
The responses adapter converts it to TextGenerationTaskParams for the pipeline.
"""

import json
from collections.abc import AsyncGenerator
from typing import Any, cast

import pydantic
import pytest

from exo.api.adapters.responses import (
    collect_responses_response,
    generate_responses_stream,
)
from exo.api.types import CompletionTokensDetails, PromptTokensDetails, Usage
from exo.api.types.openai_responses import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseInputMessage,
    ResponsesRequest,
    ResponsesResponse,
    ResponseUsage,
)
from exo.shared.types.chunks import TokenChunk
from exo.shared.types.common import CommandId, ModelId

_TEST_MODEL = ModelId("test-model")


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


class TestResponseUsage:
    """Tests for ResponseUsage with input_tokens_details and output_tokens_details."""

    def test_usage_defaults_to_zero_details(self):
        usage = ResponseUsage(
            input_tokens=10,
            input_tokens_details=InputTokensDetails(),
            output_tokens=20,
            output_tokens_details=OutputTokensDetails(),
            total_tokens=30,
        )
        assert usage.input_tokens_details.cached_tokens == 0
        assert usage.output_tokens_details.reasoning_tokens == 0

    def test_usage_with_reasoning_tokens(self):
        usage = ResponseUsage(
            input_tokens=10,
            input_tokens_details=InputTokensDetails(),
            output_tokens=20,
            output_tokens_details=OutputTokensDetails(reasoning_tokens=5),
            total_tokens=30,
        )
        assert usage.output_tokens_details.reasoning_tokens == 5

    def test_usage_with_cached_tokens(self):
        usage = ResponseUsage(
            input_tokens=10,
            input_tokens_details=InputTokensDetails(cached_tokens=7),
            output_tokens=20,
            output_tokens_details=OutputTokensDetails(),
            total_tokens=30,
        )
        assert usage.input_tokens_details.cached_tokens == 7

    def test_usage_serialization(self):
        usage = ResponseUsage(
            input_tokens=10,
            input_tokens_details=InputTokensDetails(cached_tokens=3),
            output_tokens=20,
            output_tokens_details=OutputTokensDetails(reasoning_tokens=5),
            total_tokens=30,
        )
        data = usage.model_dump()
        assert data["input_tokens_details"] == {"cached_tokens": 3}
        assert data["output_tokens_details"] == {"reasoning_tokens": 5}

    def test_usage_serialization_zero_details(self):
        usage = ResponseUsage(
            input_tokens=10,
            input_tokens_details=InputTokensDetails(),
            output_tokens=20,
            output_tokens_details=OutputTokensDetails(),
            total_tokens=30,
        )
        data = usage.model_dump()
        assert data["input_tokens_details"] == {"cached_tokens": 0}
        assert data["output_tokens_details"] == {"reasoning_tokens": 0}


def _make_usage(
    prompt_tokens: int, completion_tokens: int, reasoning_tokens: int = 0
) -> Usage:
    """Create a Usage object for testing."""
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=PromptTokensDetails(),
        completion_tokens_details=CompletionTokensDetails(
            reasoning_tokens=reasoning_tokens
        ),
    )


async def _token_chunks(
    chunks: list[TokenChunk],
) -> AsyncGenerator[TokenChunk, None]:
    for chunk in chunks:
        yield chunk


class TestCollectResponsesResponseReasoningTokens:
    """Tests for reasoning_tokens in collect_responses_response."""

    async def test_non_streaming_includes_reasoning_tokens(self):
        usage = _make_usage(10, 25, reasoning_tokens=8)
        chunks = [
            TokenChunk(
                model=_TEST_MODEL,
                token_id=0,
                text="thinking...",
                is_thinking=True,
                usage=None,
            ),
            TokenChunk(
                model=_TEST_MODEL,
                token_id=1,
                text="Hello world",
                is_thinking=False,
                usage=usage,
            ),
        ]
        command_id = CommandId("test-cmd-001")
        result_parts: list[str] = []
        async for part in collect_responses_response(
            command_id, "test-model", _token_chunks(chunks)
        ):
            result_parts.append(part)
        assert len(result_parts) == 1
        response = ResponsesResponse.model_validate_json(result_parts[0])
        assert response.usage is not None
        assert response.usage.input_tokens_details.cached_tokens == 0
        assert response.usage.output_tokens_details.reasoning_tokens == 8

    async def test_non_streaming_zero_reasoning_tokens(self):
        usage = _make_usage(10, 20, reasoning_tokens=0)
        chunks = [
            TokenChunk(
                model=_TEST_MODEL,
                token_id=0,
                text="Hello world",
                is_thinking=False,
                usage=usage,
            ),
        ]
        command_id = CommandId("test-cmd-002")
        result_parts: list[str] = []
        async for part in collect_responses_response(
            command_id, "test-model", _token_chunks(chunks)
        ):
            result_parts.append(part)
        assert len(result_parts) == 1
        response = ResponsesResponse.model_validate_json(result_parts[0])
        assert response.usage is not None
        assert response.usage.output_tokens_details.reasoning_tokens == 0
        assert response.usage.input_tokens_details.cached_tokens == 0


class TestGenerateResponsesStreamReasoningTokens:
    """Tests for reasoning_tokens in generate_responses_stream."""

    async def test_streaming_includes_reasoning_tokens(self):
        usage = _make_usage(10, 25, reasoning_tokens=8)
        chunks = [
            TokenChunk(
                model=_TEST_MODEL,
                token_id=0,
                text="thinking...",
                is_thinking=True,
                usage=None,
            ),
            TokenChunk(
                model=_TEST_MODEL,
                token_id=1,
                text="Hello world",
                is_thinking=False,
                usage=usage,
            ),
        ]
        command_id = CommandId("test-cmd-003")
        events: list[str] = []
        async for event in generate_responses_stream(
            command_id, "test-model", _token_chunks(chunks)
        ):
            events.append(event)

        # The last event should be response.completed
        last_event = events[-1]
        # Parse the SSE data
        data_line = [
            line for line in last_event.strip().split("\n") if line.startswith("data: ")
        ][0]
        data = cast(dict[str, Any], json.loads(data_line.removeprefix("data: ")))
        assert data["type"] == "response.completed"
        response_usage = cast(dict[str, Any], data["response"]["usage"])
        assert response_usage["input_tokens_details"] == {"cached_tokens": 0}
        assert response_usage["output_tokens_details"] == {"reasoning_tokens": 8}

    async def test_streaming_zero_reasoning_tokens(self):
        usage = _make_usage(10, 20, reasoning_tokens=0)
        chunks = [
            TokenChunk(
                model=_TEST_MODEL,
                token_id=0,
                text="Hello world",
                is_thinking=False,
                usage=usage,
            ),
        ]
        command_id = CommandId("test-cmd-004")
        events: list[str] = []
        async for event in generate_responses_stream(
            command_id, "test-model", _token_chunks(chunks)
        ):
            events.append(event)

        last_event = events[-1]
        data_line = [
            line for line in last_event.strip().split("\n") if line.startswith("data: ")
        ][0]
        data = cast(dict[str, Any], json.loads(data_line.removeprefix("data: ")))
        assert data["type"] == "response.completed"
        response_usage = cast(dict[str, Any], data["response"]["usage"])
        assert response_usage["input_tokens_details"] == {"cached_tokens": 0}
        assert response_usage["output_tokens_details"] == {"reasoning_tokens": 0}
