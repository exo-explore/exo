from _typeshed import Incomplete
from collections.abc import Sequence
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

logger: Incomplete

class DeepSeekV32ToolParser(ToolParser):
    prev_tool_call_arr: list[dict]
    dsml_token: str
    dsml_start_check: str
    tool_call_start_token: str
    tool_call_end_token: str
    invoke_start_prefix: str
    invoke_end_token: str
    parameter_prefix: str
    parameter_end_token: str
    current_tool_name_sent: bool
    current_tool_id: str | None
    streamed_args_for_tool: list[str]
    is_tool_call_started: bool
    failed_count: int
    current_tool_index: int
    invoke_index: int
    header_sent: bool
    current_function_name: str | None
    current_param_name: str | None
    current_param_value: str
    param_count: int
    in_param: bool
    in_function: bool
    json_started: bool
    json_closed: bool
    accumulated_params: dict
    streaming_request: ChatCompletionRequest | None
    tool_call_complete_regex: Incomplete
    invoke_complete_regex: Incomplete
    parameter_complete_regex: Incomplete
    def __init__(self, tokenizer: TokenizerLike) -> None: ...
    def adjust_request(self, request): ...
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
