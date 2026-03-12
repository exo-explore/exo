import dataclasses as dt
import enum
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
from vllm.reasoning import ReasoningParser as ReasoningParser
from vllm.tokenizers import TokenizerLike as TokenizerLike

logger: Incomplete

class Olmo3ReasoningState(enum.Enum):
    REASONING = 1
    CONTENT = 2

@dt.dataclass(frozen=True)
class Indices:
    start: int
    end: int
    def __len__(self) -> int: ...

def string_overlap(a: str, b: str) -> tuple[Indices | None, Indices | None]: ...
@dt.dataclass
class Olmo3ReasoningBuffer:
    think_start: str = ...
    think_end: str = ...
    buffer: str = ...
    state: Olmo3ReasoningState = ...
    def process_buffer(self) -> DeltaMessage | None: ...
    def __len__(self) -> int: ...
    def add_text(self, delta_text: str) -> DeltaMessage | None: ...

class Olmo3ReasoningParser(ReasoningParser):
    think_start: str
    think_end: str
    reasoning_regex: Incomplete
    buffer: Incomplete
    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs) -> None: ...
    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool: ...
    def extract_content_ids(self, input_ids: list[int]) -> list[int]: ...
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
