import abc
from _typeshed import Incomplete
from collections.abc import Sequence
from transformers import PreTrainedTokenizerBase as PreTrainedTokenizerBase
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest as ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage as DeltaMessage
from vllm.logger import init_logger as init_logger
from vllm.reasoning import ReasoningParser as ReasoningParser

logger: Incomplete

class GraniteReasoningParser(ReasoningParser, metaclass=abc.ABCMeta):
    think_start_expr: str
    response_start_expr: str
    reasoning_regex: Incomplete
    valid_think_starts: Incomplete
    valid_response_starts: Incomplete
    seq_boundary_end: str
    seq_boundary_start: str
    longest_think_start: Incomplete
    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs) -> None: ...
    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest
    ) -> tuple[str | None, str | None]: ...
    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None: ...
