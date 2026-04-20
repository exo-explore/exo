"""Tests for parse_tool_calls generator, especially unclosed tool call handling."""

import json
from collections.abc import Generator
from typing import Any

from exo.shared.types.worker.runner_response import GenerationResponse, ToolCallResponse
from exo.worker.runner.llm_inference.model_output_parsers import parse_tool_calls
from exo.worker.runner.llm_inference.tool_parsers import make_mlx_parser


def _make_responses(texts: list[str]) -> Generator[GenerationResponse]:
    """Create a sequence of GenerationResponses from text strings."""
    for i, text in enumerate(texts):
        is_last = i == len(texts) - 1
        yield GenerationResponse(
            text=text,
            token=i,
            finish_reason="stop" if is_last else None,
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
                _make_responses(texts),
                _dummy_parser,
                tools=None,
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
                tools=None,
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
                _make_responses(texts),
                make_mlx_parser("<tool_call>", "</tool_call>", _failing_parser),
                tools=None,
            )
        )

        assert len(results) == 1
        assert isinstance(results[0], GenerationResponse)
        assert results[0].text == "<tool_call>bad content</tool_call>"
        assert results[0].finish_reason == "error"

    def test_tool_schema_coerces_string_arguments_to_expected_types(self):
        """Tool argument values should be coerced using provided JSON schema."""

        def _parser_with_string_args(_text: str) -> dict[str, Any]:
            return {
                "name": "process",
                "arguments": {
                    "action": "output",
                    "id": "0",
                    "verbose": "true",
                    "temperature": "0.75",
                },
            }

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "process",
                    "description": "Manage background processes",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string"},
                            "id": {"type": "integer"},
                            "verbose": {"type": "boolean"},
                            "temperature": {"type": "number"},
                        },
                        "required": ["action"],
                    },
                },
            }
        ]

        results = list(
            parse_tool_calls(
                _make_responses(["<tool_call>", "process", "</tool_call>"]),
                make_mlx_parser(
                    "<tool_call>", "</tool_call>", _parser_with_string_args
                ),
                tools,
            )
        )

        assert len(results) == 1
        assert isinstance(results[0], ToolCallResponse)

        args = json.loads(results[0].tool_calls[0].arguments)  # pyright: ignore[reportAny]
        assert args == {
            "action": "output",
            "id": 0,
            "verbose": True,
            "temperature": 0.75,
        }

    def test_schema_coercion_skips_unknown_tools(self):
        """If no matching tool schema exists, arguments should remain unchanged."""

        def _parser_with_string_id(_text: str) -> dict[str, Any]:
            return {
                "name": "process",
                "arguments": {"action": "output", "id": "0"},
            }

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "different_tool",
                    "parameters": {
                        "type": "object",
                        "properties": {"id": {"type": "integer"}},
                    },
                },
            }
        ]

        results = list(
            parse_tool_calls(
                _make_responses(["<tool_call>", "process", "</tool_call>"]),
                make_mlx_parser("<tool_call>", "</tool_call>", _parser_with_string_id),
                tools,
            )
        )

        assert len(results) == 1
        assert isinstance(results[0], ToolCallResponse)

        args = json.loads(results[0].tool_calls[0].arguments)  # pyright: ignore[reportAny]
        assert args == {"action": "output", "id": "0"}
