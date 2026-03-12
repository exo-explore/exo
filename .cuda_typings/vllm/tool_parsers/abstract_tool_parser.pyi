from _typeshed import Incomplete
from collections.abc import Callable as Callable, Sequence
from functools import cached_property as cached_property
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest as ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage as DeltaMessage,
    ExtractedToolCallInformation as ExtractedToolCallInformation,
)
from vllm.entrypoints.openai.responses.protocol import (
    ResponseTextConfig as ResponseTextConfig,
    ResponsesRequest as ResponsesRequest,
)
from vllm.logger import init_logger as init_logger
from vllm.sampling_params import StructuredOutputsParams as StructuredOutputsParams
from vllm.tokenizers import TokenizerLike as TokenizerLike
from vllm.tool_parsers.utils import (
    get_json_schema_from_tools as get_json_schema_from_tools,
)
from vllm.utils.collection_utils import is_list_of as is_list_of
from vllm.utils.import_utils import import_from_path as import_from_path

logger: Incomplete

class ToolParser:
    prev_tool_call_arr: list[dict]
    current_tool_id: int
    current_tool_name_sent: bool
    streamed_args_for_tool: list[str]
    model_tokenizer: Incomplete
    def __init__(self, tokenizer: TokenizerLike) -> None: ...
    @cached_property
    def vocab(self) -> dict[str, int]: ...
    def adjust_request(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionRequest: ...
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

class ToolParserManager:
    tool_parsers: dict[str, type[ToolParser]]
    lazy_parsers: dict[str, tuple[str, str]]
    @classmethod
    def get_tool_parser(cls, name: str) -> type[ToolParser]: ...
    @classmethod
    def register_lazy_module(
        cls, name: str, module_path: str, class_name: str
    ) -> None: ...
    @classmethod
    def register_module(
        cls,
        name: str | list[str] | None = None,
        force: bool = True,
        module: type[ToolParser] | None = None,
    ) -> type[ToolParser] | Callable[[type[ToolParser]], type[ToolParser]]: ...
    @classmethod
    def list_registered(cls) -> list[str]: ...
    @classmethod
    def import_tool_parser(cls, plugin_path: str) -> None: ...
