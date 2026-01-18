# pyright: reportAny=false

import json

from exo.worker.parsing.stream import ChunkParser, ChunkParserConfig


def test_chunkparser_hermes_tool_call_emits_openai_style_delta() -> None:
    parser = ChunkParser(ChunkParserConfig(reasoning_parser_name=None, tool_parser_name="hermes"))

    out = parser.feed(
        '<tool_call>{"name":"do_thing","arguments":{"x":1}}</tool_call>'
    )

    assert len(out) == 1
    tool_delta = out[0]
    assert isinstance(tool_delta, dict)

    assert tool_delta["type"] == "function"
    assert tool_delta["index"] == 0

    tool_id = tool_delta["id"]
    assert isinstance(tool_id, str)
    assert tool_id.startswith("call_")

    fn = tool_delta["function"]
    assert fn["name"] == "do_thing"

    # HermesToolParser JSON-dumps arguments dict; normalize for stable comparison.
    assert json.loads(fn["arguments"]) == {"x": 1}
