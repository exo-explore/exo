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

class KimiK2ToolParser(ToolParser):
    current_tool_name_sent: bool
    prev_tool_call_arr: list[dict]
    current_tool_id: int
    streamed_args_for_tool: list[str]
    in_tool_section: bool
    token_buffer: str
    buffer_max_size: int
    section_char_count: int
    max_section_chars: int
    tool_calls_start_token: str
    tool_calls_end_token: str
    tool_calls_start_token_variants: list[str]
    tool_calls_end_token_variants: list[str]
    tool_call_start_token: str
    tool_call_end_token: str
    tool_call_regex: Incomplete
    stream_tool_call_portion_regex: Incomplete
    stream_tool_call_name_regex: Incomplete
    tool_calls_start_token_id: Incomplete
    tool_calls_end_token_id: Incomplete
    tool_calls_start_token_ids: list[int]
    tool_calls_end_token_ids: list[int]
    tool_call_start_token_id: Incomplete
    tool_call_end_token_id: Incomplete
    def __init__(self, tokenizer: TokenizerLike) -> None: ...
    def reset_streaming_state(self) -> None: ...
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
