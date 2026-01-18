from __future__ import annotations

from .function_parameter import FunctionParameterToolParser
from .hermes import HermesReasoningParser

REASONING_OPEN = "<think>"
REASONING_CLOSE = "</think>"


class Nemotron3NanoReasoningParser(HermesReasoningParser):

    def __init__(self) -> None:
        super().__init__(reasoning_open=REASONING_OPEN, reasoning_close=REASONING_CLOSE)


class Nemotron3NanoToolParser(FunctionParameterToolParser):
    pass
