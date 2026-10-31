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
        # The model's output, not a failed request: it finishes as the model finished
        assert results[0].finish_reason == "stop"

    def test_malformed_qwen_tool_call_is_returned_as_text(self):
        """What Qwen3.8 once wrote, unprompted, during a 24 h soak: its parser rejects it, and the
        request used to fail with a 500."""
        from mlx_lm.tool_parsers import qwen3_coder

        parser = make_mlx_parser(
            qwen3_coder.tool_call_start,
            qwen3_coder.tool_call_end,
            qwen3_coder.parse_tool_call,
        )
        texts = ["<tool_call>", "\n</function=\n", "</tool_call>"]
        results = list(parse_tool_calls(_make_responses(texts), parser, tools=None))

        assert len(results) == 1
        assert isinstance(results[0], GenerationResponse)
        assert results[0].text == "<tool_call>\n</function=\n</tool_call>"
        assert results[0].finish_reason == "stop"

    def test_text_after_a_failed_parse_keeps_coming(self):
        """A malformed tool call mid-answer doesn't cut off the rest of the answer."""

        def _failing_parser(text: str) -> dict[str, Any]:
            raise ValueError("parse failed")

        texts = ["<tool_call>", "</function=", "</tool_call>", " and", " more"]
        results = list(
            parse_tool_calls(
                _make_responses(texts),
                make_mlx_parser("<tool_call>", "</tool_call>", _failing_parser),
                tools=None,
            )
        )

        assert [r.text for r in results if isinstance(r, GenerationResponse)] == [
            "<tool_call></function=</tool_call>",
            " and",
            " more",
        ]
        last = results[-1]
        assert isinstance(last, GenerationResponse)
        assert last.finish_reason == "stop"

    def test_unclosed_tool_call_at_the_end_yields_text(self):
        """Generation ending inside a tool call gives back what was written, not an error."""
        texts = ["<tool_call>", "test_fn", " unfinished"]
        results = list(
            parse_tool_calls(
                _make_responses(texts),
                _dummy_parser,
                tools=None,
            )
        )

        assert len(results) == 1
        assert isinstance(results[0], GenerationResponse)
        assert results[0].text == "<tool_call>test_fn unfinished"
        assert results[0].finish_reason == "stop"

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


class TestQwenLenientToolCalls:
    """Malformed Qwen3-Coder XML calls must not come back with empty arguments."""

    def _parse(self, body: str) -> dict[str, Any]:
        from mlx_lm.tool_parsers.qwen3_coder import parse_tool_call

        parser = make_mlx_parser("<tool_call>", "</tool_call>", parse_tool_call)
        parsed = parser.parse(f"<tool_call>{body}</tool_call>", tools=None)
        assert parsed is not None
        arguments: dict[str, Any] = json.loads(parsed[0].arguments)  # pyright: ignore[reportAny]
        return arguments

    def test_missing_parameter_close(self):
        arguments = self._parse(
            "\n<function=write>\n<parameter=path>\n/a.txt\n<parameter=content>\nhi\n</function>\n"
        )
        assert arguments == {"path": "/a.txt", "content": "hi"}

    def test_json_body(self):
        arguments = self._parse(
            '\n<function=write>\n{"path": "/a.txt", "content": "hi"}\n</function>\n'
        )
        assert arguments == {"path": "/a.txt", "content": "hi"}

    def test_missing_function_close(self):
        arguments = self._parse(
            "\n<function=write>\n<parameter=path>\n/a.txt\n</parameter>\n<parameter=content>\nhi\n</parameter>\n"
        )
        assert arguments == {"path": "/a.txt", "content": "hi"}

    def test_well_formed_unchanged(self):
        arguments = self._parse(
            "\n<function=write>\n<parameter=path>\n/a.txt\n</parameter>\n<parameter=content>\nline1\n</parameter>\n</function>\n"
        )
        assert arguments == {"path": "/a.txt", "content": "line1"}

    def test_json_body_with_raw_newlines(self):
        arguments = self._parse(
            '\n<function=edit>\n{"path": "/a.py", "oldString": "def a():\n    return 1", "newString": "def a():\n    return 2"}\n</function>\n'
        )
        assert arguments == {
            "path": "/a.py",
            "oldString": "def a():\n    return 1",
            "newString": "def a():\n    return 2",
        }

    def test_parameter_name_attribute(self):
        arguments = self._parse(
            '\n<function=edit>\n<parameter name="path">/a.py</parameter>\n<parameter name="oldString">x = 1</parameter>\n<parameter name="newString">x = 2</parameter>\n</function>\n'
        )
        assert arguments == {
            "path": "/a.py",
            "oldString": "x = 1",
            "newString": "x = 2",
        }


class TestEmptyArgumentsForRequiredParameters:
    """A call that would reach the client with `{}` for a tool with required
    parameters is returned as a parse failure, so the raw text surfaces instead."""

    _tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "write",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_sequences",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]

    def _parser(self):
        from mlx_lm.tool_parsers.qwen3_coder import parse_tool_call

        return make_mlx_parser("<tool_call>", "</tool_call>", parse_tool_call)

    def test_unparseable_arguments_for_required_tool_fail(self):
        text = "<tool_call>\n<function=write>\nsome garbled body\n</function>\n</tool_call>"
        assert self._parser().parse(text, tools=self._tools) is None

    def test_no_argument_tool_still_works(self):
        text = "<tool_call>\n<function=list_sequences>\n</function>\n</tool_call>"
        parsed = self._parser().parse(text, tools=self._tools)
        assert parsed is not None
        assert parsed[0].name == "list_sequences"

    def test_valid_call_unchanged(self):
        text = "<tool_call>\n<function=write>\n<parameter=path>\n/a.py\n</parameter>\n<parameter=content>\nx = 1\n</parameter>\n</function>\n</tool_call>"
        parsed = self._parser().parse(text, tools=self._tools)
        assert parsed is not None
        assert json.loads(parsed[0].arguments) == {"path": "/a.py", "content": "x = 1"}


class TestQwenToolCallsSplitOnSchemaParameters:
    """With the tool schema known, a value runs to the next opening tag for one of
    the tool's parameters, so missing or extra close tags cannot drop or merge
    parameters."""

    _tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "edit",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filePath": {"type": "string"},
                        "oldString": {"type": "string"},
                        "newString": {"type": "string"},
                        "replaceAll": {"type": "boolean"},
                    },
                    "required": ["filePath", "oldString", "newString"],
                },
            },
        }
    ]

    def _parse(self, body: str) -> dict[str, Any]:
        from mlx_lm.tool_parsers.qwen3_coder import parse_tool_call

        parser = make_mlx_parser("<tool_call>", "</tool_call>", parse_tool_call)
        parsed = parser.parse(f"<tool_call>{body}</tool_call>", tools=self._tools)
        assert parsed is not None
        arguments: dict[str, Any] = json.loads(parsed[0].arguments)  # pyright: ignore[reportAny]
        return arguments

    def test_well_formed_call_keeps_indentation_and_types(self):
        assert self._parse(
            "\n<function=edit>\n<parameter=filePath>\n/a.py\n</parameter>\n<parameter=oldString>\n    x = 1\n\n</parameter>\n<parameter=newString>\n    x = 2\n\n</parameter>\n<parameter=replaceAll>\ntrue\n</parameter>\n</function>\n"
        ) == {
            "filePath": "/a.py",
            "oldString": "    x = 1\n",
            "newString": "    x = 2\n",
            "replaceAll": True,
        }

    def test_last_parameter_unclosed(self):
        assert self._parse(
            "\n<function=edit>\n<parameter=filePath>\n/a.py\n</parameter>\n<parameter=oldString>\nx = 1\n</parameter>\n<parameter=newString>\nx = 2\n</function>\n"
        ) == {"filePath": "/a.py", "oldString": "x = 1", "newString": "x = 2"}

    def test_middle_parameter_unclosed(self):
        assert self._parse(
            "\n<function=edit>\n<parameter=filePath>\n/a.py\n</parameter>\n<parameter=oldString>\nx = 1\n<parameter=newString>\nx = 2\n</parameter>\n</function>\n"
        ) == {"filePath": "/a.py", "oldString": "x = 1", "newString": "x = 2"}

    def test_value_containing_a_close_tag(self):
        assert self._parse(
            "\n<function=edit>\n<parameter=filePath>\n/a.md\n</parameter>\n<parameter=oldString>\nwrite `</parameter>` tags\n</parameter>\n<parameter=newString>\nclose with `</parameter>`\n</parameter>\n</function>\n"
        ) == {
            "filePath": "/a.md",
            "oldString": "write `</parameter>` tags",
            "newString": "close with `</parameter>`",
        }

    def test_misspelled_close_tag(self):
        assert self._parse(
            "\n<function=edit>\n<parameter=filePath>\n/a.py\n</parameter>\n<parameter=oldString>\nx = 1\n</parameter>\n<parameter=newString>\nx = 2\n</parameters>\n</function>\n"
        ) == {"filePath": "/a.py", "oldString": "x = 1", "newString": "x = 2"}

    def test_tag_for_an_unknown_parameter_stays_in_the_value(self):
        assert self._parse(
            "\n<function=edit>\n<parameter=filePath>\n/a.py\n</parameter>\n<parameter=oldString>\n<parameter=foo>bar</parameter>\n</parameter>\n<parameter=newString>\nok\n</parameter>\n</function>\n"
        ) == {
            "filePath": "/a.py",
            "oldString": "<parameter=foo>bar</parameter>",
            "newString": "ok",
        }
