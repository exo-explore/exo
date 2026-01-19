from __future__ import annotations

import json
from typing import Any, Callable, Dict, cast

from .abstract_parser import AbstractToolParser


def _strip_wrapping_tags(text: str, open_tag: str, close_tag: str) -> str:
    stripped = text
    if stripped.startswith(open_tag):
        stripped = stripped[len(open_tag):]
    if stripped.endswith(close_tag):
        stripped = stripped[:-len(close_tag)]
    return stripped.strip()


ParseToolCallFn = Callable[[str, Any | None], Dict[str, Any] | None]


class ToolParser(AbstractToolParser):
    """Tool-call parser built on an upstream parse function.

    This class relies on AbstractToolParser's streaming buffer (between tool_open/tool_close)
    and only invokes the upstream parser once a full block is available.
    """

    def __init__(
        self,
        *,
        tool_open: str,
        tool_close: str,
        parse_tool_call: ParseToolCallFn,
        tools: Any | None = None,
    ) -> None:
        super().__init__(tool_open=tool_open, tool_close=tool_close)
        self._parse_tool_call = parse_tool_call
        self._tools: Any | None = tools

    def extract_tool_calls(self, model_output: str) -> dict[str, Any] | None:
        inner = _strip_wrapping_tags(model_output, self.tool_open, self.tool_close)

        try:
            parsed = self._parse_tool_call(inner, self._tools)
        except Exception:
            # Treat as plain text if upstream parser rejects it.
            return {"content": model_output}

        if not isinstance(parsed, dict):
            return {"content": model_output}

        name = parsed.get("name")
        arguments = parsed.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return {"content": model_output}

        arguments_dict = cast(Dict[str, Any], arguments)

        return {
            "tool_calls": [
                {
                    "name": name,
                    "arguments": json.dumps(arguments_dict, ensure_ascii=False),
                }
            ]
        }
