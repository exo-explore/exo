from _typeshed import Incomplete
from collections.abc import Sequence
from vllm.entrypoints.chat_utils import make_tool_call_id as make_tool_call_id
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

class StreamingXMLToolCallParser:
    tools: list[ChatCompletionToolsParam] | None
    tool_call_start_token: str
    tool_call_end_token: str
    function_start_token: str
    function_end_token: str
    parameter_start_token: str
    parameter_end_token: str
    def __init__(self) -> None: ...
    deltas: Incomplete
    tool_call_index: int
    current_call_id: Incomplete
    last_completed_call_id: Incomplete
    current_function_name: Incomplete
    current_function_open: bool
    parameters: Incomplete
    current_param_name: Incomplete
    current_param_value: str
    current_param_value_converted: str
    current_param_is_first: bool
    should_emit_end_newline: bool
    start_quote_emitted: bool
    streaming_buffer: str
    last_processed_pos: int
    text_content_buffer: str
    defer_current_parameter: bool
    deferred_param_raw_value: str
    parser: Incomplete
    def reset_streaming_state(self) -> None: ...
    def parse_single_streaming_chunks(self, xml_chunk: str) -> DeltaMessage: ...
    def setup_parser(self) -> None: ...
    def set_tools(self, tools: list[ChatCompletionToolsParam] | None): ...
    def repair_param_type(self, param_type: str) -> str: ...

class Step3p5ToolParser(ToolParser):
    parser: Incomplete
    prev_tool_call_arr: list[dict]
    streamed_args_for_tool: list[str]
    def __init__(self, tokenizer: TokenizerLike) -> None: ...
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
    def parser_should_check_for_unstreamed_tool_arg_tokens(self) -> bool: ...
