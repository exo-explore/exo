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

class HunyuanA13BReasoningParser(ReasoningParser):
    think_start_expr: str
    think_end_expr: str
    response_start_expr: str
    response_end_expr: str
    full_match_reasoning_regex: Incomplete
    half_match_reasoning_regex: Incomplete
    think_start_ids: Incomplete
    think_start_ids_fast: Incomplete
    response_start_ids: Incomplete
    response_start_ids_fast: Incomplete
    response_end_ids: Incomplete
    fast_think_ids: Incomplete
    buffered_text: Incomplete
    buffered_ids: Incomplete
    current_state: str
    all_states: Incomplete
    expected_sequence: Incomplete
    expected_sequence_side: Incomplete
    sequence_index: int
    token_buffer: Incomplete
    text_buffer: str
    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs) -> None: ...
    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool: ...
    def extract_content_ids(self, input_ids: list[int]) -> list[int]: ...
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
