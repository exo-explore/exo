from _typeshed import Incomplete
from collections.abc import Sequence
from transformers import PreTrainedTokenizerBase as PreTrainedTokenizerBase
from vllm.entrypoints.mcp.tool_server import ToolServer as ToolServer
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest as ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage as DeltaMessage
from vllm.entrypoints.openai.parser.harmony_utils import (
    parse_chat_output as parse_chat_output,
)
from vllm.logger import init_logger as init_logger
from vllm.reasoning import ReasoningParser as ReasoningParser

logger: Incomplete
no_func_reaonsing_tag: Incomplete

def from_builtin_tool_to_tag(tool: str) -> list[dict]: ...
def tag_with_builtin_funcs(
    no_func_reaonsing_tag, builtin_tool_list: list[str]
) -> dict: ...

class GptOssReasoningParser(ReasoningParser):
    reasoning_end_token_ids_prefix: Incomplete
    reasoning_end_token_ids_suffix: Incomplete
    eom_token_id: Incomplete
    reasoning_max_num_between_tokens: int
    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs) -> None: ...
    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool: ...
    def extract_content_ids(self, input_ids: list[int]) -> list[int]: ...
    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None: ...
    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest
    ) -> tuple[str | None, str | None]: ...
    def prepare_structured_tag(
        self, original_tag: str | None, tool_server: ToolServer | None
    ) -> str | None: ...
