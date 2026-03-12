from _typeshed import Incomplete
from collections.abc import Sequence
from transformers import PreTrainedTokenizerBase as PreTrainedTokenizerBase
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
from vllm.tool_parsers.abstract_tool_parser import ToolParser as ToolParser
from vllm.tool_parsers.utils import (
    find_common_prefix as find_common_prefix,
    is_complete_json as is_complete_json,
    partial_json_loads as partial_json_loads,
)

logger: Incomplete

class Llama3JsonToolParser(ToolParser):
    prev_tool_call_arr: list[dict]
    current_tool_id: int
    current_tool_name_sent: bool
    streamed_args_for_tool: list[str]
    bot_token: str
    bot_token_id: Incomplete
    tool_call_start_regex: Incomplete
    json_decoder: Incomplete
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None: ...
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
