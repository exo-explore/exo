from __future__ import annotations

from .hermes import HermesReasoningParser

REASONING_OPEN = "<think>"
REASONING_CLOSE = "</think>"


class GLM4MoEReasoningParser(HermesReasoningParser):
    """Reasoning parser for GLM4 MoE model's reasoning response format."""

    def __init__(self) -> None:
        super().__init__(reasoning_open=REASONING_OPEN, reasoning_close=REASONING_CLOSE)

    def respects_enable_thinking(self) -> bool:
        return True
