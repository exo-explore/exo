from collections.abc import Iterable, Sequence
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest as ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage as DeltaMessage
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest as ResponsesRequest,
)
from vllm.reasoning.basic_parsers import (
    BaseThinkingReasoningParser as BaseThinkingReasoningParser,
)
from vllm.tokenizers import TokenizerLike as TokenizerLike

class Step3p5ReasoningParser(BaseThinkingReasoningParser):
    @property
    def start_token(self) -> str: ...
    @property
    def end_token(self) -> str: ...
    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs) -> None: ...
    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool: ...
    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool: ...
    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
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
