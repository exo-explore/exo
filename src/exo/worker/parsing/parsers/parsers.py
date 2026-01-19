from __future__ import annotations

from typing import Any, Dict, List, Optional

from mlx_lm.tool_parsers import (
    function_gemma,
    glm47,
    json_tools,
    minimax_m2,
    qwen3_coder,
)

from .tool_parser import ToolParser

ToolsSchema = Optional[List[Dict[str, Any]]]


class JsonToolsParser(ToolParser):
    def __init__(self, tools: ToolsSchema = None) -> None:
        super().__init__(
            tool_open=json_tools.tool_call_start,
            tool_close=json_tools.tool_call_end,
            parse_tool_call=json_tools.parse_tool_call,  # type: ignore[arg-type]
            tools=tools,
        )


class FunctionGemmaParser(ToolParser):
    def __init__(self, tools: ToolsSchema = None) -> None:
        super().__init__(
            tool_open=function_gemma.tool_call_start,
            tool_close=function_gemma.tool_call_end,
            parse_tool_call=function_gemma.parse_tool_call,
            tools=tools,
        )


class Glm47Parser(ToolParser):
    def __init__(self, tools: ToolsSchema = None) -> None:
        super().__init__(
            tool_open=glm47.tool_call_start,
            tool_close=glm47.tool_call_end,
            parse_tool_call=glm47.parse_tool_call,
            tools=tools,
        )


class MiniMaxM2Parser(ToolParser):
    def __init__(self, tools: ToolsSchema = None) -> None:
        super().__init__(
            tool_open=minimax_m2.tool_call_start,
            tool_close=minimax_m2.tool_call_end,
            parse_tool_call=minimax_m2.parse_tool_call,  # type: ignore[arg-type]
            tools=tools,
        )


class Qwen3CoderParser(ToolParser):
    def __init__(self, tools: ToolsSchema = None) -> None:
        super().__init__(
            tool_open=qwen3_coder.tool_call_start,
            tool_close=qwen3_coder.tool_call_end,
            parse_tool_call=qwen3_coder.parse_tool_call,
            tools=tools,
        )
