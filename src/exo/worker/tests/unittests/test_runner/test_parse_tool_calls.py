"""Tests for parse_tool_calls generator, especially unclosed tool call handling."""

from collections.abc import Generator
from typing import Any

from exo.shared.types.worker.runner_response import GenerationResponse, ToolCallResponse
from exo.worker.runner.llm_inference.runner import parse_tool_calls
from exo.worker.runner.llm_inference.tool_parsers import make_mlx_parser


def _make_responses(
    texts: list[str],
    finish_on_last: bool = True,
) -> Generator[GenerationResponse]:
    """Create a sequence of GenerationResponses from text strings."""
    for i, text in enumerate(texts):
        is_last = i == len(texts) - 1
        yield GenerationResponse(
            text=text,
            token=i,
            finish_reason="stop" if (is_last and finish_on_last) else None,
            usage=None,
        )


def _dummier_parser(text: str) -> dict[str, Any]:
    return {"name": "test_fn", "arguments": {"arg": text}}


_dummy_parser = make_mlx_parser("<tool_call>", "</tool_call>", _dummier_parser)


class TestParseToolCalls:
    """Tests for parse_tool_calls generator."""

    def test_closed_tool_call_works_normally(self):
        """Normal tool call flow should not be affected."""
        texts = ["<tool_call>", "test_fn", "</tool_call>"]
        results = list(
            parse_tool_calls(
                _make_responses(texts, finish_on_last=False),
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

    def test_failed_parse_yields_text(self):
        """When tool call parsing fails, the text should be yielded as-is."""

        def _failing_parser(text: str) -> dict[str, Any]:
            raise ValueError("parse failed")

        texts = ["<tool_call>", "bad content", "</tool_call>"]
        results = list(
            parse_tool_calls(
                _make_responses(texts, finish_on_last=False),
                make_mlx_parser("<tool_call>", "</tool_call>", _failing_parser),
            )
        )

        assert len(results) == 1
        assert isinstance(results[0], GenerationResponse)
        assert results[0].text == "<tool_call>bad content</tool_call>"
