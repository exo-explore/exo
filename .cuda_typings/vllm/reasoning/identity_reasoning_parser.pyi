from _typeshed import Incomplete
from collections.abc import Iterable, Sequence
from transformers import PreTrainedTokenizerBase as PreTrainedTokenizerBase
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest as ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage as DeltaMessage
from vllm.logger import init_logger as init_logger
from vllm.reasoning import ReasoningParser as ReasoningParser

logger: Incomplete

class IdentityReasoningParser(ReasoningParser):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs) -> None: ...
    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool: ...
    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool: ...
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
