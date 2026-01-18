from __future__ import annotations

import json
from enum import Enum
import logging

from .hermes import HermesReasoningParser
from .abstract_parser import AbstractToolParser

logger = logging.getLogger(__name__)

REASONING_OPEN = "<|think|>"
REASONING_CLOSE = "<|end|>"

CONTENT_RESPONSE_TOKEN = "<|content|>"

TOOL_OPEN = "<|tool_call:begin|>"
TOOL_CLOSE = "<|tool_call:end|>"

TOOL_NAME_PREFIX = "<|tool_call:name|>"
TOOL_ARGS_PREFIX = "<|tool_call:args|>"


class SolarOpenToolState(Enum):
    NORMAL = "normal"
    FOUND_CONTENT = "found_content"
    FOUND_TOOL_CALL = "found_tool_call"


class SolarOpenReasoningParser(HermesReasoningParser):

    def __init__(self) -> None:
        super().__init__(reasoning_open=REASONING_OPEN, reasoning_close=REASONING_CLOSE)


class SolarOpenToolParser(AbstractToolParser):

    def __init__(self) -> None:
        super().__init__(tool_open=TOOL_OPEN, tool_close=TOOL_CLOSE)
        self.state = SolarOpenToolState.NORMAL
        self.content_response_token = CONTENT_RESPONSE_TOKEN
        self.tool_name_prefix = TOOL_NAME_PREFIX
        self.tool_args_prefix = TOOL_ARGS_PREFIX

    def extract_tool_calls(self, model_output: str) -> dict[str, list] | None:
        content_idx = model_output.find(self.content_response_token)
        if content_idx != -1:
            content = model_output[content_idx + len(self.content_response_token):]
            return {"content": content}

        tool_calls = []
        remaining_output = model_output

        while self.tool_open in remaining_output:
            tool_call_open_idx = remaining_output.find(self.tool_open)
            tool_call_name_idx = remaining_output.find(self.tool_name_prefix, tool_call_open_idx)
            tool_call_args_idx = remaining_output.find(self.tool_args_prefix, tool_call_name_idx)
            tool_call_close_idx = remaining_output.find(self.tool_close, tool_call_args_idx)

            if (tool_call_name_idx == -1 or tool_call_args_idx == -1 or tool_call_close_idx == -1):
                logger.warning(
                    "Malformed tool call in output, missing required tokens: %s",
                    remaining_output[:100],
                )
                break

            tool_name = remaining_output[tool_call_name_idx + len(self.tool_name_prefix):tool_call_args_idx].strip()
            tool_args = remaining_output[tool_call_args_idx + len(self.tool_args_prefix):tool_call_close_idx].strip()

            try:
                json.loads(tool_args)
                tool_calls.append({"name": tool_name, "arguments": tool_args})
            except json.JSONDecodeError as e:
                logger.warning("Invalid JSON in tool arguments for '%s': %s", tool_name, e)

            remaining_output = remaining_output[tool_call_close_idx + len(self.tool_close):]

        return {
            "tool_calls": tool_calls,
            "content": remaining_output if remaining_output else None,
        }

    def extract_tool_calls_streaming(self, chunk: str) -> tuple[dict[str, list] | str | None, bool]:
        self.buffer += chunk

        if self.content_response_token in self.buffer:
            self.state = SolarOpenToolState.FOUND_CONTENT
            content_idx = self.buffer.find(self.content_response_token)
            content = self.buffer[content_idx + len(self.content_response_token):]
            self.buffer = ""
            return {"content": content}, True

        if self.state == SolarOpenToolState.FOUND_CONTENT:
            return {"content": chunk}, True

        if self.tool_open in self.buffer:
            self.state = SolarOpenToolState.FOUND_TOOL_CALL

        if self.state == SolarOpenToolState.FOUND_TOOL_CALL:
            if self.tool_close in self.buffer:
                result = self.extract_tool_calls(self.buffer)
                self.buffer = ""
                self.state = SolarOpenToolState.NORMAL
                return result, True
            return None, False

        return None, False
