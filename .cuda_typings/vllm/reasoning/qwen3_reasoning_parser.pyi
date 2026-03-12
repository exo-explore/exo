from _typeshed import Incomplete
from collections.abc import Sequence
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

class Qwen3ReasoningParser(BaseThinkingReasoningParser):
    thinking_enabled: Incomplete
    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs) -> None: ...
    @property
    def start_token(self) -> str: ...
    @property
    def end_token(self) -> str: ...
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
