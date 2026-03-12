import abc
from _typeshed import Incomplete
from abc import abstractmethod
from collections.abc import Sequence
from functools import cached_property as cached_property
from openai.types.responses import ResponseOutputItem as ResponseOutputItem
from openai.types.responses.response_output_text import Logprob as Logprob
from vllm.entrypoints.chat_utils import make_tool_call_id as make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam as ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest as ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage as DeltaMessage,
    ExtractedToolCallInformation as ExtractedToolCallInformation,
    FunctionCall as FunctionCall,
    FunctionDefinition as FunctionDefinition,
)
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest as ResponsesRequest,
)
from vllm.logger import init_logger as init_logger
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser as ReasoningParser
from vllm.tokenizers import TokenizerLike as TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import ToolParser as ToolParser
from vllm.utils import random_uuid as random_uuid

logger: Incomplete

class Parser(metaclass=abc.ABCMeta):
    reasoning_parser_cls: type[ReasoningParser] | None
    tool_parser_cls: type[ToolParser] | None
    model_tokenizer: Incomplete
    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs) -> None: ...
    @cached_property
    def vocab(self) -> dict[str, int]: ...
    @property
    def reasoning_parser(self) -> ReasoningParser | None: ...
    @reasoning_parser.setter
    def reasoning_parser(self, parser: ReasoningParser | None) -> None: ...
    @property
    def tool_parser(self) -> ToolParser | None: ...
    @tool_parser.setter
    def tool_parser(self, parser: ToolParser | None) -> None: ...
    @abstractmethod
    def is_reasoning_end(self, input_ids: list[int]) -> bool: ...
    def is_reasoning_end_streaming(
        self, input_ids: list[int], delta_ids: list[int]
    ) -> bool: ...
    @abstractmethod
    def extract_content_ids(self, input_ids: list[int]) -> list[int]: ...
    @abstractmethod
    def extract_response_outputs(
        self,
        model_output: str,
        request: ResponsesRequest,
        enable_auto_tools: bool = False,
        tool_call_id_type: str = "random",
        logprobs: list[Logprob] | None = None,
    ) -> list[ResponseOutputItem]: ...
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
    def adjust_request(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionRequest: ...
    @abstractmethod
    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation: ...
    @abstractmethod
    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None: ...

class DelegatingParser(Parser, metaclass=abc.ABCMeta):
    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> tuple[str | None, str | None]: ...
    def extract_response_outputs(
        self,
        model_output: str,
        request: ResponsesRequest,
        enable_auto_tools: bool = False,
        tool_call_id_type: str = "random",
        logprobs: list[Logprob] | None = None,
    ) -> list[ResponseOutputItem]: ...
    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None: ...
    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation: ...
    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None: ...

class _WrappedParser(DelegatingParser, metaclass=abc.ABCMeta):
    reasoning_parser_cls: type[ReasoningParser] | None
    tool_parser_cls: type[ToolParser] | None
    def __init__(self, tokenizer: TokenizerLike) -> None: ...
