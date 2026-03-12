from _typeshed import Incomplete
from collections.abc import Sequence
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest as ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage as DeltaMessage
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest as ResponsesRequest,
)
from vllm.logger import init_logger as init_logger
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser as ReasoningParser
from vllm.reasoning.basic_parsers import (
    BaseThinkingReasoningParser as BaseThinkingReasoningParser,
)
from vllm.tokenizers import TokenizerLike as TokenizerLike

logger: Incomplete

class MiniMaxM2ReasoningParser(BaseThinkingReasoningParser):
    @property
    def start_token(self) -> str: ...
    @property
    def end_token(self) -> str: ...
    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None: ...

class MiniMaxM2AppendThinkReasoningParser(ReasoningParser):
    end_token_id: Incomplete
    start_token_id: Incomplete
    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs) -> None: ...
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
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> tuple[str | None, str | None]: ...
