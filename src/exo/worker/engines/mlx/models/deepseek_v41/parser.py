from collections.abc import Generator

from exo.api.types import ToolCallItem
from exo.shared.types.worker.runner_response import GenerationResponse, ToolCallResponse
from exo.worker.engines.mlx.models.deepseek_v3.parser import parse_dsml_stream

# DeepSeek V4.1's DSML tool-call tags have a space before the tag name (V4's do not).
DSML_TOKEN = "｜DSML｜"
TOOL_CALLS_START = f"<{DSML_TOKEN} calls>"
TOOL_CALLS_END = f"</{DSML_TOKEN} calls>"


def parse_tool_calls(text: str) -> list[ToolCallItem] | None:
    """The tool calls of one `<｜DSML｜ calls>` block, parsed by DeepSeek's encoder; None
    if the block is malformed."""
    from exo.worker.engines.mlx.models.deepseek_v41.encoding import (
        parse_tool_calls as parse_dsml,
    )

    start = text.find(TOOL_CALLS_START)
    if start < 0:
        return None
    # the encoder reads from just after the block's name
    index = start + len(TOOL_CALLS_START) - 1
    try:
        parsed = parse_dsml(index, text)
    except ValueError:
        return None
    items: list[ToolCallItem] = []
    for call in parsed[2]:
        name = call["name"]
        namespace = call.get("namespace")
        if namespace is not None:
            name = f"{namespace}::{name}"
        items.append(ToolCallItem(name=name, arguments=call["arguments"]))
    return items


def parse_deepseek_v41(
    responses: Generator[GenerationResponse | None],
) -> Generator[GenerationResponse | ToolCallResponse | None]:
    return parse_dsml_stream(
        responses, TOOL_CALLS_START, TOOL_CALLS_END, parse_tool_calls
    )
