import abc
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Callable as Callable
from contextlib import AsyncExitStack
from mcp.client import ClientSession as ClientSession
from openai.types.responses.tool import Mcp as Mcp
from openai_harmony import Message
from typing import Final
from vllm import envs as envs
from vllm.entrypoints.chat_utils import (
    ChatTemplateContentFormatOption as ChatTemplateContentFormatOption,
)
from vllm.entrypoints.constants import MCP_PREFIX as MCP_PREFIX
from vllm.entrypoints.mcp.tool import Tool as Tool
from vllm.entrypoints.mcp.tool_server import ToolServer as ToolServer
from vllm.entrypoints.openai.engine.protocol import FunctionCall as FunctionCall
from vllm.entrypoints.openai.parser.harmony_utils import (
    get_encoding as get_encoding,
    get_streamable_parser_for_assistant as get_streamable_parser_for_assistant,
    render_for_completion as render_for_completion,
)
from vllm.entrypoints.openai.parser.responses_parser import (
    get_responses_parser_for_simple_context as get_responses_parser_for_simple_context,
)
from vllm.entrypoints.openai.responses.protocol import (
    ResponseInputOutputItem as ResponseInputOutputItem,
    ResponseRawMessageAndToken as ResponseRawMessageAndToken,
    ResponsesRequest as ResponsesRequest,
)
from vllm.entrypoints.openai.responses.utils import (
    construct_tool_dicts as construct_tool_dicts,
)
from vllm.outputs import RequestOutput as RequestOutput
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser as ReasoningParser
from vllm.tokenizers import TokenizerLike as TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import ToolParser as ToolParser
from vllm.utils import random_uuid as random_uuid

logger: Incomplete

class TurnMetrics:
    input_tokens: Incomplete
    output_tokens: Incomplete
    cached_input_tokens: Incomplete
    tool_output_tokens: Incomplete
    def __init__(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
        tool_output_tokens: int = 0,
    ) -> None: ...
    def reset(self) -> None: ...
    def copy(self) -> TurnMetrics: ...

class ConversationContext(ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    def append_output(self, output: RequestOutput) -> None: ...
    @abstractmethod
    def append_tool_output(self, output) -> None: ...
    @abstractmethod
    async def call_tool(self) -> list[Message]: ...
    @abstractmethod
    def need_builtin_tool_call(self) -> bool: ...
    @abstractmethod
    def render_for_completion(self) -> list[int]: ...
    @abstractmethod
    async def init_tool_sessions(
        self,
        tool_server: ToolServer | None,
        exit_stack: AsyncExitStack,
        request_id: str,
        mcp_tools: dict[str, Mcp],
    ) -> None: ...
    @abstractmethod
    async def cleanup_session(self) -> None: ...

class SimpleContext(ConversationContext):
    last_output: Incomplete
    num_prompt_tokens: int
    num_output_tokens: int
    num_cached_tokens: int
    num_reasoning_tokens: int
    all_turn_metrics: Incomplete
    input_messages: list[ResponseRawMessageAndToken]
    def __init__(self) -> None: ...
    def append_output(self, output) -> None: ...
    @property
    def output_messages(self) -> list[ResponseRawMessageAndToken]: ...
    @property
    def final_output(self) -> RequestOutput | None: ...
    def append_tool_output(self, output) -> None: ...
    def need_builtin_tool_call(self) -> bool: ...
    async def call_tool(self) -> list[Message]: ...
    def render_for_completion(self) -> list[int]: ...
    async def init_tool_sessions(
        self,
        tool_server: ToolServer | None,
        exit_stack: AsyncExitStack,
        request_id: str,
        mcp_tools: dict[str, Mcp],
    ) -> None: ...
    async def cleanup_session(self) -> None: ...

class ParsableContext(ConversationContext):
    num_prompt_tokens: int
    num_output_tokens: int
    num_cached_tokens: int
    num_reasoning_tokens: int
    all_turn_metrics: list[TurnMetrics]
    parser: Incomplete
    tool_parser_cls: Incomplete
    request: Incomplete
    available_tools: Incomplete
    called_tools: set[str]
    tool_dicts: Incomplete
    chat_template: Incomplete
    chat_template_content_format: Final[Incomplete]
    input_messages: list[ResponseRawMessageAndToken]
    output_messages: list[ResponseRawMessageAndToken]
    def __init__(
        self,
        *,
        response_messages: list[ResponseInputOutputItem],
        tokenizer: TokenizerLike,
        reasoning_parser_cls: Callable[[TokenizerLike], ReasoningParser] | None,
        request: ResponsesRequest,
        available_tools: list[str] | None,
        tool_parser_cls: Callable[[TokenizerLike], ToolParser] | None,
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
    ) -> None: ...
    def append_output(self, output: RequestOutput) -> None: ...
    def append_tool_output(self, output: list[ResponseInputOutputItem]) -> None: ...
    def need_builtin_tool_call(self) -> bool: ...
    async def call_python_tool(
        self, tool_session: ClientSession | Tool, last_msg: FunctionCall
    ) -> list[ResponseInputOutputItem]: ...
    async def call_search_tool(
        self, tool_session: ClientSession | Tool, last_msg: FunctionCall
    ) -> list[ResponseInputOutputItem]: ...
    async def call_container_tool(
        self, tool_session: ClientSession | Tool, last_msg: Message
    ) -> list[Message]: ...
    async def call_tool(self) -> list[ResponseInputOutputItem]: ...
    def render_for_completion(self) -> None: ...
    async def init_tool_sessions(
        self,
        tool_server: ToolServer | None,
        exit_stack: AsyncExitStack,
        request_id: str,
        mcp_tools: dict[str, Mcp],
    ): ...
    async def cleanup_session(self, *args, **kwargs) -> None: ...

class HarmonyContext(ConversationContext):
    finish_reason: str | None
    available_tools: Incomplete
    called_tools: set[str]
    parser: Incomplete
    num_init_messages: Incomplete
    num_prompt_tokens: int
    num_output_tokens: int
    num_cached_tokens: int
    num_reasoning_tokens: int
    num_tool_output_tokens: int
    current_turn_metrics: Incomplete
    all_turn_metrics: list[TurnMetrics]
    is_first_turn: bool
    first_tok_of_message: bool
    def __init__(self, messages: list, available_tools: list[str]) -> None: ...
    def append_output(self, output: RequestOutput) -> None: ...
    def append_tool_output(self, output: list[Message]) -> None: ...
    @property
    def messages(self) -> list: ...
    def need_builtin_tool_call(self) -> bool: ...
    async def call_tool(self) -> list[Message]: ...
    def render_for_completion(self) -> list[int]: ...
    async def call_search_tool(
        self, tool_session: ClientSession | Tool, last_msg: Message
    ) -> list[Message]: ...
    async def call_python_tool(
        self, tool_session: ClientSession | Tool, last_msg: Message
    ) -> list[Message]: ...
    async def init_tool_sessions(
        self,
        tool_server: ToolServer | None,
        exit_stack: AsyncExitStack,
        request_id: str,
        mcp_tools: dict[str, Mcp],
    ): ...
    async def call_container_tool(
        self, tool_session: ClientSession | Tool, last_msg: Message
    ) -> list[Message]: ...
    async def cleanup_session(self, *args, **kwargs) -> None: ...

class StreamingHarmonyContext(HarmonyContext):
    last_output: Incomplete
    parser: Incomplete
    encoding: Incomplete
    last_tok: Incomplete
    first_tok_of_message: bool
    last_content_delta: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def messages(self) -> list: ...
    def append_output(self, output: RequestOutput) -> None: ...
    def append_tool_output(self, output: list[Message]) -> None: ...
    def is_expecting_start(self) -> bool: ...
    def is_assistant_action_turn(self) -> bool: ...
    def render_for_completion(self) -> list[int]: ...
