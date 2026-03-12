import abc
from _typeshed import Incomplete
from abc import abstractmethod
from collections.abc import Callable as Callable, Iterable, Sequence
from functools import cached_property as cached_property
from vllm.entrypoints.mcp.tool_server import ToolServer as ToolServer
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest as ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage as DeltaMessage
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest as ResponsesRequest,
)
from vllm.logger import init_logger as init_logger
from vllm.tokenizers import TokenizerLike as TokenizerLike
from vllm.utils.collection_utils import is_list_of as is_list_of
from vllm.utils.import_utils import import_from_path as import_from_path

logger: Incomplete

class ReasoningParser(metaclass=abc.ABCMeta):
    model_tokenizer: Incomplete
    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs) -> None: ...
    @cached_property
    def vocab(self) -> dict[str, int]: ...
    @abstractmethod
    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool: ...
    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool: ...
    @abstractmethod
    def extract_content_ids(self, input_ids: list[int]) -> list[int]: ...
    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int: ...
    @abstractmethod
    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> tuple[str | None, str | None]: ...
    @abstractmethod
    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None: ...
    def prepare_structured_tag(
        self, original_tag: str | None, tool_server: ToolServer | None
    ) -> str | None: ...

class ReasoningParserManager:
    reasoning_parsers: dict[str, type[ReasoningParser]]
    lazy_parsers: dict[str, tuple[str, str]]
    @classmethod
    def get_reasoning_parser(cls, name: str) -> type[ReasoningParser]: ...
    @classmethod
    def list_registered(cls) -> list[str]: ...
    @classmethod
    def register_lazy_module(
        cls, name: str, module_path: str, class_name: str
    ) -> None: ...
    @classmethod
    def register_module(
        cls,
        name: str | list[str] | None = None,
        force: bool = True,
        module: type[ReasoningParser] | None = None,
    ) -> (
        type[ReasoningParser] | Callable[[type[ReasoningParser]], type[ReasoningParser]]
    ): ...
    @classmethod
    def import_reasoning_parser(cls, plugin_path: str) -> None: ...
