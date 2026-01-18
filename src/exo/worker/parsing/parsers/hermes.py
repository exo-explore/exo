from __future__ import annotations

import re
import json

from .abstract_parser import (
    AbstractReasoningParser,
    AbstractToolParser,
    ReasoningParserState,
)

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
REASONING_OPEN = "<think>"
REASONING_CLOSE = "</think>"


class HermesReasoningParser(AbstractReasoningParser):

    def __init__(
        self,
        reasoning_open: str = REASONING_OPEN,
        reasoning_close: str = REASONING_CLOSE
    ) -> None:
        super().__init__(reasoning_open=reasoning_open,
                         reasoning_close=reasoning_close)
        self.reasoning_regex = re.compile(
            f"{re.escape(reasoning_open)}(.*?){re.escape(reasoning_close)}",
            re.DOTALL
        )

    def extract_reasoning(self, model_output: str) -> dict[str, str] | None:
        matches = self.reasoning_regex.findall(model_output)
        after_reasoning_close_content = None
        if not matches:
            return {"content": model_output}
        reasoning_content_end_idx = model_output.rfind(self.reasoning_close)
        after_reasoning_close_content = model_output[reasoning_content_end_idx + len(self.reasoning_close):]
        return {
            "reasoning_content": matches[0],
            "after_reasoning_close_content": after_reasoning_close_content
        }

    def extract_reasoning_streaming(self, chunk: str) -> tuple[dict[str, str] | str | None, bool]:
        if self.reasoning_open in chunk:
            self.state = ReasoningParserState.FOUND_PREFIX
            reasoning_content_start_idx = chunk.find(self.reasoning_open)
            reasoning_content = chunk[reasoning_content_start_idx + len(self.reasoning_open):]
            return {"reasoning_content": reasoning_content}, False

        if self.state == ReasoningParserState.FOUND_PREFIX:
            if self.reasoning_close in chunk:
                reasoning_content_end_idx = chunk.find(self.reasoning_close)
                reasoning_content = chunk[:reasoning_content_end_idx]
                after_reasoning_close_content = chunk[reasoning_content_end_idx + len(self.reasoning_close):]
                return {
                    "reasoning_content": reasoning_content,
                    "after_reasoning_close_content":
                    after_reasoning_close_content
                }, True
            return {"reasoning_content": chunk}, False

        return {"content": chunk}, False


class HermesToolParser(AbstractToolParser):

    def __init__(
        self,
        tool_open: str = TOOL_OPEN,
        tool_close: str = TOOL_CLOSE
    ) -> None:
        super().__init__(tool_open=tool_open, tool_close=tool_close)
        self.tool_regex = re.compile(
            f"{re.escape(tool_open)}(.*?){re.escape(tool_close)}",
            re.DOTALL
        )

    def extract_tool_calls(self, model_output: str) -> dict[str, list] | None:
        matches = self.tool_regex.findall(model_output)
        if not matches:
            return {"content": model_output}
        tool_calls = []
        for match in matches:
            try:
                tool_data = json.loads(match.strip())
                tool_calls.append({
                    "name":
                    tool_data.get("name", ""),
                    "arguments":
                    json.dumps(tool_data.get("arguments", {})),
                })
            except json.JSONDecodeError:
                continue
        return {"tool_calls": tool_calls}
