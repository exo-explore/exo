# pyright: reportAny=false
"""Tests asserting OpenAI-spec wire shape for /v1/chat/completions deltas."""

import json
from collections.abc import AsyncGenerator
from typing import Any

from exo.api.adapters.chat_completions import (
    collect_chat_response,
    generate_chat_stream,
)
from exo.api.types import (
    CompletionTokensDetails,
    PromptTokensDetails,
    ToolCallItem,
    Usage,
)
from exo.shared.types.chunks import (
    ErrorChunk,
    PrefillProgressChunk,
    TokenChunk,
    ToolCallChunk,
)
from exo.shared.types.common import CommandId, ModelId

_TEST_MODEL = ModelId("test-model")
_NULLABLE_DELTA_FIELDS = {"content", "refusal"}


def _make_usage(prompt_tokens: int = 1, completion_tokens: int = 1) -> Usage:
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=PromptTokensDetails(),
        completion_tokens_details=CompletionTokensDetails(),
    )


async def _stream(
    chunks: list[PrefillProgressChunk | ErrorChunk | ToolCallChunk | TokenChunk],
) -> AsyncGenerator[
    PrefillProgressChunk | ErrorChunk | ToolCallChunk | TokenChunk, None
]:
    for chunk in chunks:
        yield chunk


def _parse_data_events(lines: list[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in lines:
        for sub in line.split("\n"):
            if sub.startswith("data: ") and not sub.endswith("[DONE]"):
                events.append(json.loads(sub[len("data: ") :]))
    return events


def _assert_delta_spec_compliant(delta: dict[str, Any]) -> None:
    """Reject any null delta key the OpenAI spec doesn't allow to be null."""
    for key, value in delta.items():
        if value is None and key not in _NULLABLE_DELTA_FIELDS:
            raise AssertionError(
                f"delta.{key} is null but spec requires it to be absent or a value; "
                f"full delta={delta!r}"
            )


class TestTokenStreamDeltaShape:
    async def test_token_chunk_delta_has_no_disallowed_nulls(self):
        chunks: list[PrefillProgressChunk | ErrorChunk | ToolCallChunk | TokenChunk] = [
            TokenChunk(
                model=_TEST_MODEL,
                token_id=1,
                text="Hello",
                usage=None,
            ),
            TokenChunk(
                model=_TEST_MODEL,
                token_id=2,
                text=" world",
                usage=_make_usage(),
                finish_reason="stop",
            ),
        ]
        lines: list[str] = []
        async for event in generate_chat_stream(
            CommandId("test-cmd-token"), _stream(chunks)
        ):
            lines.append(event)

        events = _parse_data_events(lines)
        assert len(events) == 2
        for event in events:
            delta = event["choices"][0]["delta"]
            _assert_delta_spec_compliant(delta)
            assert "tool_calls" not in delta or isinstance(delta["tool_calls"], list)
            assert "function_call" not in delta
            assert "name" not in delta
            assert "tool_call_id" not in delta

    async def test_thinking_chunk_delta_has_no_disallowed_nulls(self):
        chunks: list[PrefillProgressChunk | ErrorChunk | ToolCallChunk | TokenChunk] = [
            TokenChunk(
                model=_TEST_MODEL,
                token_id=1,
                text="Hmm",
                usage=None,
                is_thinking=True,
            ),
        ]
        lines: list[str] = []
        async for event in generate_chat_stream(
            CommandId("test-cmd-thinking"), _stream(chunks)
        ):
            lines.append(event)

        events = _parse_data_events(lines)
        assert len(events) == 1
        delta = events[0]["choices"][0]["delta"]
        _assert_delta_spec_compliant(delta)
        assert delta.get("reasoning_content") == "Hmm"
        assert "content" not in delta


class TestToolCallStreamDeltaShape:
    async def test_tool_call_chunk_delta_has_array_tool_calls(self):
        chunks: list[PrefillProgressChunk | ErrorChunk | ToolCallChunk | TokenChunk] = [
            ToolCallChunk(
                model=_TEST_MODEL,
                tool_calls=[
                    ToolCallItem(id="call_1", name="get_weather", arguments="{}"),
                ],
                usage=_make_usage(),
            ),
        ]
        lines: list[str] = []
        async for event in generate_chat_stream(
            CommandId("test-cmd-tool"), _stream(chunks)
        ):
            lines.append(event)

        events = _parse_data_events(lines)
        assert len(events) == 1
        delta = events[0]["choices"][0]["delta"]
        _assert_delta_spec_compliant(delta)
        assert isinstance(delta["tool_calls"], list)
        assert delta["tool_calls"][0]["function"]["name"] == "get_weather"


class TestErrorStreamShape:
    async def test_error_chunk_response_has_no_nulls(self):
        chunks: list[PrefillProgressChunk | ErrorChunk | ToolCallChunk | TokenChunk] = [
            ErrorChunk(model=_TEST_MODEL, error_message="boom"),
        ]
        lines: list[str] = []
        async for event in generate_chat_stream(
            CommandId("test-cmd-err"), _stream(chunks)
        ):
            lines.append(event)

        events = _parse_data_events(lines)
        assert len(events) == 1
        assert events[0]["error"]["message"] == "boom"
        for value in events[0]["error"].values():
            assert value is not None


class TestNonStreamingResponseShape:
    async def test_collected_response_message_has_no_disallowed_nulls(self):
        chunks: list[PrefillProgressChunk | ErrorChunk | ToolCallChunk | TokenChunk] = [
            TokenChunk(
                model=_TEST_MODEL,
                token_id=1,
                text="Hello",
                usage=_make_usage(),
                finish_reason="stop",
            ),
        ]
        parts: list[str] = []
        async for part in collect_chat_response(
            CommandId("test-cmd-nonstream"), _stream(chunks)
        ):
            parts.append(part)

        assert len(parts) == 1
        payload = json.loads(parts[0])
        message = payload["choices"][0]["message"]
        for key, value in message.items():
            if value is None:
                assert key in {"content", "refusal", "reasoning_content"}, (
                    f"non-streaming message.{key} is null but spec disallows it"
                )
        assert "function_call" not in message
        assert "name" not in message
        assert "tool_call_id" not in message
