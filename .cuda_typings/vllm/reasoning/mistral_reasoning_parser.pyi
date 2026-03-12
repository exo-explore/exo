from _typeshed import Incomplete
from collections.abc import Sequence
from functools import cached_property as cached_property
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest as ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest as ResponsesRequest,
)
from vllm.logger import init_logger as init_logger
from vllm.reasoning import ReasoningParser as ReasoningParser
from vllm.reasoning.basic_parsers import (
    BaseThinkingReasoningParser as BaseThinkingReasoningParser,
)
from vllm.tokenizers.mistral import MistralTokenizer as MistralTokenizer

logger: Incomplete

class MistralReasoningParser(BaseThinkingReasoningParser):
    start_token_id: Incomplete
    end_token_id: Incomplete
    def __init__(self, tokenizer: MistralTokenizer, *args, **kwargs) -> None: ...
    @cached_property
    def start_token(self) -> str: ...
    @cached_property
    def end_token(self) -> str: ...
    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool: ...
    def extract_content_ids(self, input_ids: list[int]) -> list[int]: ...
    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> tuple[str | None, str | None]: ...
