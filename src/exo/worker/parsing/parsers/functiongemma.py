from __future__ import annotations

import re
import json

from .abstract_parser import AbstractToolParser

TOOL_OPEN = "<start_function_call>"
TOOL_CLOSE = "<end_function_call>"


class FunctionGemmaToolParser(AbstractToolParser):

    def __init__(
        self,
        tool_open: str = TOOL_OPEN,
        tool_close: str = TOOL_CLOSE
    ) -> None:
        super().__init__(tool_open=tool_open, tool_close=tool_close)
        self.tool_call_regex = re.compile(
            r"<start_function_call>call:(\w+)\{(.*?)\}<end_function_call>"
            r"|<start_function_call>call:(\w+)\{(.*)",
            re.DOTALL,
        )
        self.arg_regex = re.compile(
            r"(\w+):<escape>(.*?)<escape>",
            re.DOTALL,
        )

    def extract_tool_calls(self, model_output: str) -> dict[str, list] | None:
        matches = self.tool_call_regex.findall(model_output)
        if not matches:
            return {"content": model_output}
        tool_calls = []
        for match in matches:
            function_name = match[0]
            args_str = match[1]
            args_matches = self.arg_regex.findall(args_str)
            args_dict = {key: value for key, value in args_matches}
            tool_calls.append({
                "name": function_name,
                "arguments": json.dumps(args_dict)
            })
        return {"tool_calls": tool_calls}
