from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .abstract_parser import (
    AbstractReasoningParser,
    AbstractToolParser,
    ReasoningParserState,
    ToolParserState,
)
from .function_parameter import FunctionParameterToolParser
from .functiongemma import FunctionGemmaToolParser
from .glm4_moe import GLM4MoEReasoningParser, GLM4MoEToolParser
from .harmony import HarmonyParser
from .hermes import HermesReasoningParser, HermesToolParser
from .minimax_m2 import MiniMaxM2ReasoningParser, MiniMaxM2ToolParser
from .nemotron3_nano import Nemotron3NanoReasoningParser, Nemotron3NanoToolParser
from .qwen3 import Qwen3ReasoningParser, Qwen3ToolParser
from .qwen3_coder import Qwen3CoderToolParser
from .qwen3_moe import Qwen3MoEReasoningParser, Qwen3MoEToolParser
from .qwen3_vl import Qwen3VLReasoningParser, Qwen3VLToolParser
from .iquest_coder_v1 import IQuestCoderV1ToolParser
from .solar_open import SolarOpenReasoningParser, SolarOpenToolParser

REASONING_PARSER_MAP: dict[str, type[AbstractReasoningParser]] = {
    "hermes": HermesReasoningParser,
    "qwen3": Qwen3ReasoningParser,
    "qwen3_moe": Qwen3MoEReasoningParser,
    "qwen3_vl": Qwen3VLReasoningParser,
    "glm4_moe": GLM4MoEReasoningParser,
    "minimax_m2": MiniMaxM2ReasoningParser,
    "nemotron3_nano": Nemotron3NanoReasoningParser,
    "solar_open": SolarOpenReasoningParser,
}

TOOL_PARSER_MAP: dict[str, type[AbstractToolParser]] = {
    "hermes": HermesToolParser,
    "qwen3": Qwen3ToolParser,
    "qwen3_coder": Qwen3CoderToolParser,
    "qwen3_moe": Qwen3MoEToolParser,
    "qwen3_vl": Qwen3VLToolParser,
    "glm4_moe": GLM4MoEToolParser,
    "minimax_m2": MiniMaxM2ToolParser,
    "nemotron3_nano": Nemotron3NanoToolParser,
    "functiongemma": FunctionGemmaToolParser,
    "iquest_coder_v1": IQuestCoderV1ToolParser,
    "solar_open": SolarOpenToolParser,
    "function_parameter": FunctionParameterToolParser,
}

UNIFIED_PARSER_MAP: dict[str, type] = {
    "harmony": HarmonyParser,
}


def get_reasoning_parser(parser_name: str | None) -> type[AbstractReasoningParser] | None:
    if parser_name is None:
        return None
    return REASONING_PARSER_MAP.get(parser_name.lower())


def get_tool_parser(parser_name: str | None) -> type[AbstractToolParser] | None:
    if parser_name is None:
        return None
    return TOOL_PARSER_MAP.get(parser_name.lower())


def get_unified_parser(parser_name: str | None) -> type | None:
    if parser_name is None:
        return None
    return UNIFIED_PARSER_MAP.get(parser_name.lower())


@dataclass
class ParsersResult:
    reasoning_parser: AbstractReasoningParser | None = None
    tool_parser: AbstractToolParser | None = None
    unified_parser: Any | None = None
    parser_name: str | None = None

    @property
    def is_unified(self) -> bool:
        return self.unified_parser is not None

    @property
    def has_reasoning(self) -> bool:
        return self.reasoning_parser is not None or self.unified_parser is not None

    @property
    def has_tool_parsing(self) -> bool:
        return self.tool_parser is not None or self.unified_parser is not None


class ParserManager:

    @staticmethod
    def create_parsers(
        reasoning_parser_name: str | None = None,
        tool_parser_name: str | None = None,
    ) -> ParsersResult:
        result = ParsersResult()

        reasoning_name = reasoning_parser_name.lower() if reasoning_parser_name else None
        tool_name = tool_parser_name.lower() if tool_parser_name else None

        unified_name = ParserManager._get_unified_parser_name(reasoning_name, tool_name)
        if unified_name:
            parser_class = UNIFIED_PARSER_MAP[unified_name]
            result.unified_parser = parser_class()
            result.parser_name = unified_name
            return result

        if reasoning_name and reasoning_name in REASONING_PARSER_MAP:
            result.reasoning_parser = REASONING_PARSER_MAP[reasoning_name]()
            result.parser_name = reasoning_name

        if tool_name and tool_name in TOOL_PARSER_MAP:
            result.tool_parser = TOOL_PARSER_MAP[tool_name]()
            if not result.parser_name:
                result.parser_name = tool_name

        return result

    @staticmethod
    def _get_unified_parser_name(
        reasoning_name: str | None,
        tool_name: str | None,
    ) -> str | None:
        if (reasoning_name and tool_name and reasoning_name == tool_name and reasoning_name in UNIFIED_PARSER_MAP):
            return reasoning_name

        if reasoning_name and reasoning_name in UNIFIED_PARSER_MAP:
            return reasoning_name
        if tool_name and tool_name in UNIFIED_PARSER_MAP:
            return tool_name

        return None

    @staticmethod
    def is_unified_parser(parser_name: str | None) -> bool:
        if not parser_name:
            return False
        return parser_name.lower() in UNIFIED_PARSER_MAP


__all__ = [
    "AbstractReasoningParser",
    "AbstractToolParser",
    "ReasoningParserState",
    "ToolParserState",
    "HermesReasoningParser",
    "Qwen3ReasoningParser",
    "Qwen3MoEReasoningParser",
    "Qwen3VLReasoningParser",
    "GLM4MoEReasoningParser",
    "MiniMaxM2ReasoningParser",
    "Nemotron3NanoReasoningParser",
    "HermesToolParser",
    "Qwen3ToolParser",
    "Qwen3CoderToolParser",
    "Qwen3MoEToolParser",
    "Qwen3VLToolParser",
    "GLM4MoEToolParser",
    "MiniMaxM2ToolParser",
    "Nemotron3NanoToolParser",
    "FunctionGemmaToolParser",
    "FunctionParameterToolParser",
    "HarmonyParser",
    "REASONING_PARSER_MAP",
    "TOOL_PARSER_MAP",
    "UNIFIED_PARSER_MAP",
    "get_reasoning_parser",
    "get_tool_parser",
    "get_unified_parser",
    "ParserManager",
    "ParsersResult",
]
