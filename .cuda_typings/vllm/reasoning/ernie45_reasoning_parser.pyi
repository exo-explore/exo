from _typeshed import Incomplete
from collections.abc import Sequence
from transformers import PreTrainedTokenizerBase as PreTrainedTokenizerBase
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest as ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage as DeltaMessage
from vllm.logger import init_logger as init_logger
from vllm.reasoning.basic_parsers import (
    BaseThinkingReasoningParser as BaseThinkingReasoningParser,
)

logger: Incomplete

class Ernie45ReasoningParser(BaseThinkingReasoningParser):
    response_start_token: str
    response_end_token: str
    newline_token: str
    @property
    def start_token(self) -> str: ...
    @property
    def end_token(self) -> str: ...
    start_token_id: Incomplete
    end_token_id: Incomplete
    response_start_token_id: Incomplete
    response_end_token_id: Incomplete
    newline_token_id: Incomplete
    parser_token_ids: Incomplete
    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs) -> None: ...
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
