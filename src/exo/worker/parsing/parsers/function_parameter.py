from __future__ import annotations

import json
import re

from .abstract_parser import AbstractToolParser

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"


class FunctionParameterToolParser(AbstractToolParser):

    def __init__(
        self,
        tool_open: str = TOOL_OPEN,
        tool_close: str = TOOL_CLOSE,
    ) -> None:
        super().__init__(tool_open=tool_open, tool_close=tool_close)
        self.tool_regex = re.compile(
            r"<function=([^>]+)>\s*(.*?)\s*</function>",
            re.DOTALL,
        )
        self.parameter_regex = re.compile(
            r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
            re.DOTALL,
        )

    def extract_tool_calls(self, model_output: str) -> dict[str, list] | None:
        matches = self.tool_regex.findall(model_output)
        if not matches:
            return {"content": model_output}

        tool_calls = []
        for match in matches:
            function_name = match[0].strip()
            function_content = match[1].strip()

            param_matches = self.parameter_regex.findall(function_content)
            arguments: dict[str, object] = {}
            for param_match in param_matches:
                param_name = param_match[0].strip()
                param_value = param_match[1].strip()
                try:
                    arguments[param_name] = json.loads(param_value)
                except (json.JSONDecodeError, ValueError):
                    arguments[param_name] = param_value

            tool_calls.append({
                "name": function_name,
                "arguments": json.dumps(arguments),
            })

        return {"tool_calls": tool_calls}
