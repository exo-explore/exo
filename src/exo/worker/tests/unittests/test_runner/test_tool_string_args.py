"""A tool argument the schema declares as a string comes back as a string, even when its text
is JSON that the model's tool parser decoded."""

import json
from collections.abc import Generator
from typing import Any

import pytest
from mlx_lm.tool_parsers.glm47 import parse_tool_call as glm47_parse_tool_call

from exo.shared.types.worker.runner_response import (
    GenerationResponse,
    ToolCallResponse,
)
from exo.worker.runner.llm_inference.model_output_parsers import parse_tool_calls
from exo.worker.runner.llm_inference.tool_parsers import (
    _coerce_tool_arg_with_schema,  # pyright: ignore[reportPrivateUsage]
    make_mlx_parser,
)

STRING: dict[str, Any] = {"type": "string"}


@pytest.mark.parametrize(
    ("value", "text"),
    [
        ({"name": "demo", "version": 1}, '{"name": "demo", "version": 1}'),
        ([1, "two", None], '[1, "two", null]'),
        (42, "42"),
        (1.5, "1.5"),
        (True, "true"),
        ({"greeting": "héllo ✓"}, '{"greeting": "héllo ✓"}'),
    ],
)
def test_string_schema_turns_decoded_values_back_into_json_text(
    value: object, text: str
) -> None:
    assert _coerce_tool_arg_with_schema(value, STRING) == text


def test_string_schema_leaves_strings_and_null_alone() -> None:
    assert _coerce_tool_arg_with_schema('{"a": 1}', STRING) == '{"a": 1}'
    assert _coerce_tool_arg_with_schema("plain", STRING) == "plain"
    assert _coerce_tool_arg_with_schema(None, STRING) is None


@pytest.mark.parametrize(
    ("schema", "value", "expected"),
    [
        ({"type": "integer"}, "5", 5),
        ({"type": "integer"}, {"a": 1}, {"a": 1}),
        ({"type": "number"}, "2.5", 2.5),
        ({"type": "boolean"}, "true", True),
        ({"type": "object"}, '{"a": 1}', {"a": 1}),
        ({"type": "array"}, "[1, 2]", [1, 2]),
        ({"type": "object"}, 7, 7),
        ({}, {"a": 1}, {"a": 1}),
    ],
)
def test_other_schema_types_are_unchanged(
    schema: dict[str, Any], value: object, expected: object
) -> None:
    assert _coerce_tool_arg_with_schema(value, schema) == expected


@pytest.mark.parametrize(
    ("types", "value", "expected"),
    [
        (["string", "null"], None, None),
        (["string", "null"], {"a": 1}, '{"a": 1}'),
        (["string"], [1, 2], "[1, 2]"),
        # another allowed type takes the value as it is: no text
        (["string", "object"], {"a": 1}, {"a": 1}),
        (["object", "string"], {"a": 1}, {"a": 1}),
        (["string", "integer"], 3, 3),
        # existing order-dependent coercion of strings is kept
        (["integer", "string"], "5", 5),
        (["string", "integer"], "5", "5"),
        # nothing else takes it: text
        (["integer", "string"], {"a": 1}, '{"a": 1}'),
    ],
)
def test_unions_turn_a_value_into_text_only_when_no_other_type_takes_it(
    types: list[str], value: object, expected: object
) -> None:
    assert _coerce_tool_arg_with_schema(value, {"type": types}) == expected


def test_nested_string_properties_and_items() -> None:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "config": STRING,
            "count": {"type": "integer"},
            "tags": {"type": "array", "items": STRING},
            "inner": {"type": "object", "properties": {"body": STRING}},
        },
    }
    value = {
        "config": {"debug": True},
        "count": "3",
        "tags": [1, {"x": 2}, "plain"],
        "inner": {"body": [1, 2]},
        "extra": {"left": "alone"},
    }
    assert _coerce_tool_arg_with_schema(value, schema) == {
        "config": '{"debug": true}',
        "count": 3,
        "tags": ["1", '{"x": 2}', "plain"],
        "inner": {"body": "[1, 2]"},
        "extra": {"left": "alone"},
    }


WRITE_FILE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": STRING,
                    "content": STRING,
                    "mode": {"type": "integer"},
                },
            },
        },
    }
]
# A file's contents as the model writes them: pretty-printed JSON.
DOCUMENT = '{\n  "name": "demo",\n  "version": 1.0,\n  "tags": ["a", "b"]\n}'


def _glm_call(content: str) -> str:
    return (
        "<tool_call>write_file"
        "<arg_key>path</arg_key><arg_value>demo.json</arg_value>"
        f"<arg_key>content</arg_key><arg_value>{content}</arg_value>"
        "<arg_key>mode</arg_key><arg_value>420</arg_value>"
        "</tool_call>"
    )


def test_glm_style_call_keeps_the_json_document_as_written() -> None:
    parser = make_mlx_parser("<tool_call>", "</tool_call>", glm47_parse_tool_call)
    calls = parser.parse(_glm_call(DOCUMENT), tools=WRITE_FILE_TOOLS)
    assert calls is not None and len(calls) == 1
    arguments = json.loads(calls[0].arguments)  # pyright: ignore[reportAny]
    # the model's text, whitespace and number spelling included
    assert arguments == {"path": "demo.json", "content": DOCUMENT, "mode": 420}


def test_parser_without_the_tools_writes_decoded_values_back_as_json_text() -> None:
    def decoding_parser(text: str) -> dict[str, Any]:
        # a parser that decodes argument values and can't be given the tools
        return {
            "name": "write_file",
            "arguments": {
                "path": "demo.json",
                "content": {"name": "demo"},
                "mode": 420,
            },
        }

    parser = make_mlx_parser("<tool_call>", "</tool_call>", decoding_parser)
    calls = parser.parse("<tool_call>x</tool_call>", tools=WRITE_FILE_TOOLS)
    assert calls is not None
    arguments = json.loads(calls[0].arguments)  # pyright: ignore[reportAny]
    assert arguments == {
        "path": "demo.json",
        "content": '{"name": "demo"}',
        "mode": 420,
    }


def test_without_tools_nothing_is_coerced() -> None:
    parser = make_mlx_parser("<tool_call>", "</tool_call>", glm47_parse_tool_call)
    calls = parser.parse(_glm_call(DOCUMENT), tools=None)
    assert calls is not None
    arguments = json.loads(calls[0].arguments)  # pyright: ignore[reportAny]
    assert arguments["content"] == json.loads(DOCUMENT)


def test_streamed_glm_style_call_through_the_runner_parser() -> None:
    parser = make_mlx_parser("<tool_call>", "</tool_call>", glm47_parse_tool_call)
    body = _glm_call(DOCUMENT).removeprefix("<tool_call>").removesuffix("</tool_call>")
    # token by token, as the runner sees it: the markers are tokens of their own
    pieces = ["<tool_call>", *(body[i : i + 7] for i in range(0, len(body), 7))]
    pieces.append("</tool_call>")

    def responses() -> Generator[GenerationResponse | None]:
        for i, piece in enumerate(pieces):
            yield GenerationResponse.model_validate(
                {
                    "text": piece,
                    "token": i,
                    "finish_reason": "stop" if i == len(pieces) - 1 else None,
                    "usage": None,
                }
            )

    results = list(parse_tool_calls(responses(), parser, tools=WRITE_FILE_TOOLS))
    tool_calls = [r for r in results if isinstance(r, ToolCallResponse)]
    assert len(tool_calls) == 1
    (item,) = tool_calls[0].tool_calls
    assert item.name == "write_file"
    assert json.loads(item.arguments) == {
        "path": "demo.json",
        "content": DOCUMENT,
        "mode": 420,
    }
