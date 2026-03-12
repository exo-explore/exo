import abc
from _typeshed import Incomplete
from vllm.logger import init_logger as init_logger
from vllm.parser.abstract_parser import DelegatingParser as DelegatingParser
from vllm.reasoning.minimax_m2_reasoning_parser import (
    MiniMaxM2ReasoningParser as MiniMaxM2ReasoningParser,
)
from vllm.tokenizers import TokenizerLike as TokenizerLike
from vllm.tool_parsers.minimax_m2_tool_parser import (
    MinimaxM2ToolParser as MinimaxM2ToolParser,
)

logger: Incomplete

class MiniMaxM2Parser(DelegatingParser, metaclass=abc.ABCMeta):
    reasoning_parser_cls = MiniMaxM2ReasoningParser
    tool_parser_cls = MinimaxM2ToolParser
    def __init__(self, tokenizer: TokenizerLike) -> None: ...
