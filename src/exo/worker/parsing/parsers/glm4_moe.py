from __future__ import annotations

import re
import json

from .abstract_parser import AbstractToolParser
from .hermes import HermesReasoningParser

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
REASONING_OPEN = "<think>"
REASONING_CLOSE = "</think>"


class GLM4MoEReasoningParser(HermesReasoningParser):
    """Reasoning parser for GLM4 MoE model's reasoning response format.

    Handles the GLM4 MoE model's reasoning response format:
    <think>reasoning_content</think>
    """

    def __init__(self) -> None:
        """Initialize the Hermes4 reasoning parser with appropriate regex patterns."""
        super().__init__(reasoning_open=REASONING_OPEN, reasoning_close=REASONING_CLOSE)
    
    def respects_enable_thinking(self) -> bool:
        """Check if the reasoning parser respects the enable_thinking flag.
        
        Returns
        -------
        bool
            True if the reasoning parser respects the enable_thinking flag, False otherwise.
        """
        return True


class GLM4MoEToolParser(AbstractToolParser):
    """Tool parser for GLM4 MoE model's tool response format.

    Handles the GLM4 MoE model's tool response format:
    <tool_call>{function_name}
    <arg_key>{arg-key-1}</arg_key>
    <arg_value>{arg-value-1}</arg_value>
    <arg_key>{arg-key-2}</arg_key>
    <arg_value>{arg-value-2}</arg_value>
    ...
    <arg_key>{arg-key-n}</arg_key>
    <arg_value>{arg-value-n}</arg_value>
    </tool_call>
    """

    def __init__(self, tool_open: str = TOOL_OPEN, tool_close: str = TOOL_CLOSE) -> None:
        """Initialize the GLM4 MoE tool parser with appropriate regex patterns."""
        super().__init__(tool_open=tool_open, tool_close=tool_close)
        
        self.func_call_regex = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
        self.func_detail_regex = re.compile(
            r"<tool_call>(.*?)(<arg_key>.*?)?</tool_call>", re.DOTALL
        )
        self.func_arg_regex = re.compile(
            r"<arg_key>(.*?)</arg_key>(?:\\n|\s)*<arg_value>(.*?)</arg_value>",
            re.DOTALL,
        )

    def extract_tool_calls(self, model_output: str) -> dict[str, list] | None:
        """Extract tool calls from complete model output.

        Parameters
        ----------
        model_output : str
            Complete model output containing tool calls in JSON format.

        Returns
        -------
        dict[str, list] | None
            Dictionary with 'tool_calls' key containing list of parsed tool calls,
            or None if no tool calls found. Each tool call has 'name' and 'arguments'.
        """
        matches = self.func_call_regex.findall(model_output)
        if not matches:
            return {"content": model_output}
        
        tool_calls = []
        for match in matches:
            tc_detail = self.func_detail_regex.search(match)
            tc_name = tc_detail.group(1)
            tc_args = tc_detail.group(2)
            pairs = self.func_arg_regex.findall(tc_args)
            arg_dct = {}
            for key, value in pairs:
                arg_key = key.strip()
                arg_value = value.strip()
                arg_dct[arg_key] = arg_value
            tool_calls.append({
                "name": tc_name.strip(),
                "arguments": json.dumps(arg_dct, ensure_ascii=False)
            })
        
        return {"tool_calls": tool_calls}
