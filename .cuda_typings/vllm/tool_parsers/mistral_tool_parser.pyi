import ijson
from _typeshed import Incomplete
from collections.abc import Generator, Sequence
from enum import Enum
from typing import Any
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
from vllm.utils.mistral import is_mistral_tokenizer as is_mistral_tokenizer

logger: Incomplete
ALPHANUMERIC: Incomplete

class StreamingState(Enum):
    WAITING_FOR_TOOL_START = ...
    WAITING_FOR_TOOL_KEY = ...
    PARSING_NAME = ...
    PARSING_NAME_COMPLETED = ...
    WAITING_FOR_ARGUMENTS_START = ...
    PARSING_ARGUMENTS = ...
    PARSING_ARGUMENTS_COMPLETED = ...
    TOOL_COMPLETE = ...
    ALL_TOOLS_COMPLETE = ...

class MistralToolCall(ToolCall):
    id: str
    @staticmethod
    def generate_random_id(): ...
    @staticmethod
    def is_valid_id(id: str) -> bool: ...

class MistralToolParser(ToolParser):
    prev_tool_call_arr: list[dict[str, Any]]
    current_tool_id: int
    streaming_state: StreamingState
    current_tool_name: str | None
    current_tool_mistral_id: str | None
    starting_new_tool: bool
    parse_coro: Incomplete
    bot_token: str
    bot_token_id: Incomplete
    tool_call_regex: Incomplete
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
    @ijson.coroutine
    def update_stream_state_pre_v11_tokenizer(self) -> Generator[None, Incomplete]: ...
