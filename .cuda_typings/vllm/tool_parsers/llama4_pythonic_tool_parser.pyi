from _typeshed import Incomplete
from collections.abc import Sequence
from transformers import PreTrainedTokenizerBase as PreTrainedTokenizerBase
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest as ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage as DeltaMessage,
    ExtractedToolCallInformation as ExtractedToolCallInformation,
)
from vllm.logger import init_logger as init_logger
from vllm.tool_parsers.abstract_tool_parser import ToolParser as ToolParser
from vllm.tool_parsers.utils import (
    UnexpectedAstError as UnexpectedAstError,
    compute_tool_delta as compute_tool_delta,
    handle_single_tool as handle_single_tool,
    make_valid_python as make_valid_python,
)

logger: Incomplete

class Llama4PythonicToolParser(ToolParser):
    TOOL_CALL_REGEX: Incomplete
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None: ...
    @property
    def current_tool_index(self) -> int: ...
    current_tool_id: Incomplete
    @current_tool_index.setter
    def current_tool_index(self, value: int) -> None: ...
    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation: ...
    prev_tool_call_arr: Incomplete
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
