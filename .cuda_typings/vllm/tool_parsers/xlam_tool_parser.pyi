from _typeshed import Incomplete
from collections.abc import Sequence
from typing import Any
from vllm.entrypoints.chat_utils import make_tool_call_id as make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest as ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall as DeltaFunctionCall,
    DeltaMessage as DeltaMessage,
    DeltaToolCall as DeltaToolCall,
    ExtractedToolCallInformation as ExtractedToolCallInformation,
    FunctionCall as FunctionCall,
    ToolCall as ToolCall,
)
from vllm.logger import init_logger as init_logger
from vllm.tokenizers import TokenizerLike as TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import ToolParser as ToolParser
from vllm.utils import random_uuid as random_uuid

logger: Incomplete

class xLAMToolParser(ToolParser):
    prev_tool_calls: list[dict]
    current_tool_id: int
    current_tool_name_sent: bool
    streamed_args: list[str]
    current_tools_sent: list[bool]
    prev_tool_call_arr: Incomplete
    json_code_block_patterns: Incomplete
    thinking_tag_pattern: str
    streaming_state: dict[str, Any]
    def __init__(self, tokenizer: TokenizerLike) -> None: ...
    def preprocess_model_output(
        self, model_output: str
    ) -> tuple[str | None, str | None]: ...
    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation: ...
    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None: ...
