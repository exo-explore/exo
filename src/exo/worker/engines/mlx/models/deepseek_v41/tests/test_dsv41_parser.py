"""DeepSeek V4.1's tool calls (DSML with spaced tag names) are parsed from the stream."""

from collections.abc import Generator

from exo.shared.types.worker.runner_response import GenerationResponse, ToolCallResponse
from exo.worker.engines.mlx.models.deepseek_v41.parser import (
    parse_deepseek_v41,
    parse_tool_calls,
)

CALL = (
    "\n\n<｜DSML｜ calls>\n"
    '<｜DSML｜ invoke name="get_weather">\n'
    '<｜DSML｜ parameter name="city" string="true">Rome</｜DSML｜ parameter>\n'
    '<｜DSML｜ parameter name="days" string="false">3</｜DSML｜ parameter>\n'
    "</｜DSML｜ invoke>\n"
    '<｜DSML｜ invoke name="search::web">\n'
    '<｜DSML｜ parameter name="q" string="true">rome weather</｜DSML｜ parameter>\n'
    "</｜DSML｜ invoke>\n"
    "</｜DSML｜ calls>"
)


def _stream(pieces: list[str]) -> Generator[GenerationResponse | None]:
    for i, piece in enumerate(pieces):
        yield GenerationResponse(
            text=piece,
            token=i,
            finish_reason="stop" if i == len(pieces) - 1 else None,
            usage=None,
        )


def test_parses_calls_with_their_argument_types_and_namespaces() -> None:
    items = parse_tool_calls(CALL)
    assert items is not None
    assert [(item.name, item.arguments) for item in items] == [
        ("get_weather", '{"city": "Rome", "days": 3}'),
        ("search::web", '{"q": "rome weather"}'),
    ]


def test_unspaced_v4_tags_are_not_v41_calls() -> None:
    assert parse_tool_calls(CALL.replace("｜DSML｜ ", "｜DSML｜")) is None


def test_stream_with_text_then_calls() -> None:
    pieces = [
        "Let me check.",
        "\n\n<｜DSML｜",
        " calls>\n",
        CALL.split(" calls>\n", 1)[1],
        "",
    ]
    out = [r for r in parse_deepseek_v41(_stream(pieces)) if r is not None]
    texts = [r.text for r in out if isinstance(r, GenerationResponse)]
    calls = [r for r in out if isinstance(r, ToolCallResponse)]
    assert "".join(texts).strip() == "Let me check."
    assert len(calls) == 1
    assert [item.name for item in calls[0].tool_calls] == ["get_weather", "search::web"]


def test_malformed_calls_come_back_as_text() -> None:
    broken = "<｜DSML｜ calls>\n<｜DSML｜ invoke nam=1>\n</｜DSML｜ calls>"
    out = [r for r in parse_deepseek_v41(_stream([broken])) if r is not None]
    assert all(isinstance(r, GenerationResponse) for r in out)
    assert "".join(r.text for r in out if isinstance(r, GenerationResponse)) == broken
