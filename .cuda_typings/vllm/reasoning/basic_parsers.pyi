import abc
from _typeshed import Incomplete
from abc import abstractmethod
from collections.abc import Iterable, Sequence
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest as ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage as DeltaMessage
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest as ResponsesRequest,
)
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser as ReasoningParser
from vllm.tokenizers import TokenizerLike as TokenizerLike

class BaseThinkingReasoningParser(ReasoningParser, metaclass=abc.ABCMeta):
    @property
    @abstractmethod
    def start_token(self) -> str: ...
    @property
    @abstractmethod
    def end_token(self) -> str: ...
    start_token_id: Incomplete
    end_token_id: Incomplete
    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs) -> None: ...
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
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> tuple[str | None, str | None]: ...
    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int: ...
