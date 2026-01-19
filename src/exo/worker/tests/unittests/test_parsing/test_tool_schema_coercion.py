# pyright: reportAny=false

import json

from exo.worker.parsing.stream import ChunkParser, ChunkParserConfig


def test_qwen3_coder_schema_coercion() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "myfunc",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer"},
                        "flag": {"type": "boolean"},
                        "obj": {"type": "object"},
                        "arr": {"type": "array"},
                    },
                },
            },
        }
    ]

    parser = ChunkParser(
        ChunkParserConfig(
            reasoning_parser_name=None,
            tool_parser_name="qwen3_coder",
            tools=tools,
        )
    )

    out = parser.feed(
        "<tool_call>"
        "<function=myfunc>"
        "<parameter=x>1</parameter>"
        "<parameter=flag>true</parameter>"
        "<parameter=obj>{\"a\": 1}</parameter>"
        "<parameter=arr>[1, 2]</parameter>"
        "</function>"
        "</tool_call>"
    )

    assert len(out) == 1
    tool_delta = out[0]
    assert isinstance(tool_delta, dict)
    assert tool_delta["type"] == "function"

    args = json.loads(tool_delta["function"]["arguments"])
    assert args == {"x": 1, "flag": True, "obj": {"a": 1}, "arr": [1, 2]}


def test_minimax_m2_schema_coercion() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "myfunc",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer"},
                        "flag": {"type": "boolean"},
                        "items": {"type": "array"},
                        "maybe": {"type": ["string", "null"]},
                    },
                },
            },
        }
    ]

    parser = ChunkParser(
        ChunkParserConfig(
            reasoning_parser_name=None,
            tool_parser_name="minimax_m2",
            tools=tools,
        )
    )

    out = parser.feed(
        "<minimax:tool_call>"
        "<invoke name=\"myfunc\">"
        "<parameter name=\"x\">1</parameter>"
        "<parameter name=\"flag\">true</parameter>"
        "<parameter name=\"items\">[1, 2]</parameter>"
        "<parameter name=\"maybe\">null</parameter>"
        "</invoke>"
        "</minimax:tool_call>"
    )

    assert len(out) == 1
    tool_delta = out[0]
    assert isinstance(tool_delta, dict)
    assert tool_delta["type"] == "function"

    args = json.loads(tool_delta["function"]["arguments"])
    assert args == {"x": 1, "flag": True, "items": [1, 2], "maybe": None}
