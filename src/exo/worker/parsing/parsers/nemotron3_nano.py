from __future__ import annotations

from .hermes import HermesReasoningParser

REASONING_OPEN = "<think>"
REASONING_CLOSE = "</think>"


class Nemotron3NanoReasoningParser(HermesReasoningParser):

    def __init__(self) -> None:
        super().__init__(reasoning_open=REASONING_OPEN, reasoning_close=REASONING_CLOSE)
