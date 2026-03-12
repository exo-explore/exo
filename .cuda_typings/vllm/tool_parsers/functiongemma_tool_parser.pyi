from _typeshed import Incomplete
from collections.abc import Sequence
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

logger: Incomplete

class FunctionGemmaToolParser(ToolParser):
    current_tool_name_sent: bool
    prev_tool_call_arr: list[dict]
    current_tool_id: int
    streamed_args_for_tool: list[str]
    tool_call_start_token: str
    tool_call_end_token: str
    tool_call_regex: Incomplete
    arg_regex: Incomplete
    tool_call_start_token_ids: Incomplete
    tool_call_end_token_ids: Incomplete
    buffered_delta_text: str
    def __init__(self, tokenizer: TokenizerLike) -> None: ...
    def adjust_request(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionRequest: ...
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
