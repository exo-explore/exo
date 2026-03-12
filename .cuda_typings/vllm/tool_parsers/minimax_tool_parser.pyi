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
from vllm.tool_parsers.utils import (
    extract_intermediate_diff as extract_intermediate_diff,
)

logger: Incomplete

class MinimaxToolParser(ToolParser):
    streaming_state: dict[str, Any]
    tool_call_start_token: str
    tool_call_end_token: str
    tool_call_regex: Incomplete
    thinking_tag_pattern: str
    tool_name_pattern: Incomplete
    tool_args_pattern: Incomplete
    pending_buffer: str
    in_thinking_tag: bool
    tool_call_start_token_id: Incomplete
    tool_call_end_token_id: Incomplete
    def __init__(self, tokenizer: TokenizerLike) -> None: ...
    def preprocess_model_output(self, model_output: str) -> str: ...
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
