from __future__ import annotations

import re

from .hermes import HermesReasoningParser
from .glm4_moe import GLM4MoEToolParser

TOOL_OPEN = "<minimax:tool_call>"
TOOL_CLOSE = "</minimax:tool_call>"
REASONING_OPEN = "<think>"
REASONING_CLOSE = "</think>"


class MiniMaxM2ReasoningParser(HermesReasoningParser):

    def __init__(self) -> None:
        super().__init__(reasoning_open=REASONING_OPEN, reasoning_close=REASONING_CLOSE)

    def needs_redacted_reasoning_prefix(self) -> bool:
        return True


class MiniMaxM2ToolParser(GLM4MoEToolParser):

    def __init__(
        self,
        tool_open: str = TOOL_OPEN,
        tool_close: str = TOOL_CLOSE
    ) -> None:
        super().__init__(tool_open=tool_open, tool_close=tool_close)

        self.func_call_regex = re.compile(
            r"<minimax:tool_call>(.*?)</minimax:tool_call>", re.DOTALL)
        self.func_detail_regex = re.compile(
            r'<invoke name="([^"]+)"\s*>(.*)',
            re.DOTALL
        )
        self.func_arg_regex = re.compile(
            r'<parameter name="([^"]+)"\s*>([^<]*)</parameter>',
            re.DOTALL
        )
