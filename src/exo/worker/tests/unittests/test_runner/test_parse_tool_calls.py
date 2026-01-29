"""Tests for parse_tool_calls generator, especially unclosed tool call handling."""

from collections.abc import Generator
from typing import Any

from exo.shared.types.worker.runner_response import GenerationResponse, ToolCallResponse
from exo.worker.runner.runner import parse_tool_calls


def _make_responses(
    texts: list[str],
    finish_on_last: bool = True,
) -> Generator[GenerationResponse | ToolCallResponse]:
    """Create a sequence of GenerationResponses from text strings."""
    for i, text in enumerate(texts):
        is_last = i == len(texts) - 1
        yield GenerationResponse(
            text=text,
            token=i,
            finish_reason="stop" if (is_last and finish_on_last) else None,
        )


def _dummy_parser(text: str) -> dict[str, Any]:
    return {"name": "test_fn", "arguments": {"arg": text}}


class TestParseToolCallsUnclosedToolCall:
    """Tests for when <tool_call> is opened but </tool_call> never arrives."""

    def test_unclosed_tool_call_flushes_buffered_text(self):
        """Generator exhausting inside an unclosed tool call should yield the buffered text."""
        texts = ["Hello ", "<tool_call>", "some ", "tool ", "content"]
        results = list(
            parse_tool_calls(
                _make_responses(texts),
                "<tool_call>",
                "</tool_call>",
                _dummy_parser,
            )
        )

        # Should get the "Hello " text and then the flushed tool call text
        assert len(results) == 2
        assert isinstance(results[0], GenerationResponse)
        assert results[0].text == "Hello "
        assert results[0].finish_reason is None

        assert isinstance(results[1], GenerationResponse)
        assert results[1].text == "<tool_call>some tool content"
        assert results[1].finish_reason == "stop"

    def test_unclosed_tool_call_preserves_finish_reason(self):
        """The finish_reason from the last buffered token should be preserved."""
        texts = ["<tool_call>", "buffered"]
        results = list(
            parse_tool_calls(
                _make_responses(texts, finish_on_last=True),
                "<tool_call>",
                "</tool_call>",
                _dummy_parser,
            )
        )

        assert len(results) == 1
        assert isinstance(results[0], GenerationResponse)
        assert results[0].finish_reason == "stop"
        assert results[0].text == "<tool_call>buffered"

    def test_closed_tool_call_works_normally(self):
        """Normal tool call flow should not be affected."""
        texts = ["<tool_call>", "test_fn", "</tool_call>"]
        results = list(
            parse_tool_calls(
                _make_responses(texts, finish_on_last=False),
                "<tool_call>",
                "</tool_call>",
                _dummy_parser,
            )
        )

        assert len(results) == 1
        assert isinstance(results[0], ToolCallResponse)

    def test_no_tool_call_passes_through(self):
        """Responses without tool calls should pass through unchanged."""
        texts = ["Hello", " world"]
        results = list(
            parse_tool_calls(
                _make_responses(texts),
                "<tool_call>",
                "</tool_call>",
                _dummy_parser,
            )
        )

        assert len(results) == 2
        assert all(isinstance(r, GenerationResponse) for r in results)
        r0 = results[0]
        r1 = results[1]
        assert isinstance(r0, GenerationResponse)
        assert isinstance(r1, GenerationResponse)
        assert r0.text == "Hello"
        assert r1.text == " world"
        assert r1.finish_reason == "stop"

    def test_unclosed_tool_call_only_start_token(self):
        """<tool_call> as the only/last token should still yield flushed text."""
        texts = ["<tool_call>"]
        results = list(
            parse_tool_calls(
                _make_responses(texts, finish_on_last=True),
                "<tool_call>",
                "</tool_call>",
                _dummy_parser,
            )
        )

        assert len(results) == 1
        assert isinstance(results[0], GenerationResponse)
        assert results[0].text == "<tool_call>"
        assert results[0].finish_reason == "stop"

    def test_unclosed_tool_call_finish_on_start_token(self):
        """finish_reason on <tool_call> start token propagates to flushed response."""

        # Simulate: model outputs text, then <tool_call> with finish_reason,
        # then more text without finish_reason
        def _responses() -> Generator[GenerationResponse | ToolCallResponse]:
            yield GenerationResponse(text="Hi ", token=0, finish_reason=None)
            yield GenerationResponse(text="<tool_call>", token=1, finish_reason="stop")
            yield GenerationResponse(text="data", token=2, finish_reason=None)

        results = list(
            parse_tool_calls(
                _responses(),
                "<tool_call>",
                "</tool_call>",
                _dummy_parser,
            )
        )

        assert len(results) == 2
        assert isinstance(results[0], GenerationResponse)
        assert results[0].text == "Hi "
        assert isinstance(results[1], GenerationResponse)
        assert results[1].text == "<tool_call>data"
        # finish_reason from start token should propagate since buffered tokens lack it
        assert results[1].finish_reason == "stop"

    def test_failed_parse_yields_text(self):
        """When tool call parsing fails, the text should be yielded as-is."""

        def _failing_parser(text: str) -> dict[str, Any]:
            raise ValueError("parse failed")

        texts = ["<tool_call>", "bad content", "</tool_call>"]
        results = list(
            parse_tool_calls(
                _make_responses(texts, finish_on_last=False),
                "<tool_call>",
                "</tool_call>",
                _failing_parser,
            )
        )

        assert len(results) == 1
        assert isinstance(results[0], GenerationResponse)
        assert results[0].text == "<tool_call>bad content</tool_call>"
