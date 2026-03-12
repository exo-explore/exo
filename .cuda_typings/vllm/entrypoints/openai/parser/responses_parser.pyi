from _typeshed import Incomplete
from collections.abc import Callable as Callable
from openai.types.responses import ResponseOutputItem as ResponseOutputItem
from vllm.entrypoints.constants import MCP_PREFIX as MCP_PREFIX
from vllm.entrypoints.openai.responses.protocol import (
    ResponseInputOutputItem as ResponseInputOutputItem,
    ResponsesRequest as ResponsesRequest,
)
from vllm.outputs import CompletionOutput as CompletionOutput
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser as ReasoningParser
from vllm.tokenizers import TokenizerLike as TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import ToolParser as ToolParser
from vllm.utils import random_uuid as random_uuid

logger: Incomplete

class ResponsesParser:
    response_messages: list[ResponseInputOutputItem]
    num_init_messages: Incomplete
    tokenizer: Incomplete
    request: Incomplete
    reasoning_parser_instance: Incomplete
    tool_parser_instance: Incomplete
    finish_reason: str | None
    def __init__(
        self,
        *,
        tokenizer: TokenizerLike,
        reasoning_parser_cls: Callable[[TokenizerLike], ReasoningParser],
        response_messages: list[ResponseInputOutputItem],
        request: ResponsesRequest,
        tool_parser_cls: Callable[[TokenizerLike], ToolParser] | None,
    ) -> None: ...
    def process(self, output: CompletionOutput) -> ResponsesParser: ...
    def make_response_output_items_from_parsable_context(
        self,
    ) -> list[ResponseOutputItem]: ...

def get_responses_parser_for_simple_context(
    *,
    tokenizer: TokenizerLike,
    reasoning_parser_cls: Callable[[TokenizerLike], ReasoningParser],
    response_messages: list[ResponseInputOutputItem],
    request: ResponsesRequest,
    tool_parser_cls,
) -> ResponsesParser: ...
