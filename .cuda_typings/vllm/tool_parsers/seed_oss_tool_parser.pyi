from _typeshed import Incomplete
from collections.abc import Sequence
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest as ChatCompletionRequest,
    ChatCompletionToolsParam as ChatCompletionToolsParam,
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

logger: Incomplete

class SeedOssToolParser(ToolParser):
    TOOL_CALL_START: str
    TOOL_CALL_END: str
    prev_tool_call_arr: list[dict]
    tool_call_start_token: str
    tool_call_end_token: str
    tool_call_prefix: str
    function_end_token: str
    parameter_prefix: str
    parameter_end_token: str
    think_start_token: str
    think_end_token: str
    is_tool_call_started: bool
    is_thinking_end: bool
    failed_count: int
    tool_call_start_token_id: Incomplete
    tool_call_end_token_id: Incomplete
    think_end_token_id: Incomplete
    tool_call_complete_regex: Incomplete
    tool_call_regex: Incomplete
    tool_call_function_regex: Incomplete
    tool_call_parameter_regex: Incomplete
    def __init__(self, tokenizer: TokenizerLike) -> None: ...
    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation: ...
    accumulated_text: Incomplete
    header_sent: bool
    param_count: int
    json_started: bool
    json_closed: bool
    current_function_name: Incomplete
    current_tool_id: Incomplete
    in_function: bool
    current_param_name: Incomplete
    in_param: bool
    current_param_value: str
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
