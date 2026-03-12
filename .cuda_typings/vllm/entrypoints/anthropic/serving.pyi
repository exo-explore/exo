from _typeshed import Incomplete
from collections.abc import AsyncGenerator
from fastapi import Request as Request
from vllm.engine.protocol import EngineClient as EngineClient
from vllm.entrypoints.anthropic.protocol import (
    AnthropicContentBlock as AnthropicContentBlock,
    AnthropicContextManagement as AnthropicContextManagement,
    AnthropicCountTokensRequest as AnthropicCountTokensRequest,
    AnthropicCountTokensResponse as AnthropicCountTokensResponse,
    AnthropicDelta as AnthropicDelta,
    AnthropicError as AnthropicError,
    AnthropicMessagesRequest as AnthropicMessagesRequest,
    AnthropicMessagesResponse as AnthropicMessagesResponse,
    AnthropicStreamEvent as AnthropicStreamEvent,
    AnthropicUsage as AnthropicUsage,
)
from vllm.entrypoints.chat_utils import (
    ChatTemplateContentFormatOption as ChatTemplateContentFormatOption,
)
from vllm.entrypoints.logger import RequestLogger as RequestLogger
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam as ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest as ChatCompletionRequest,
    ChatCompletionResponse as ChatCompletionResponse,
    ChatCompletionStreamResponse as ChatCompletionStreamResponse,
    ChatCompletionToolsParam as ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.chat_completion.serving import (
    OpenAIServingChat as OpenAIServingChat,
)
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse as ErrorResponse,
    StreamOptions as StreamOptions,
)
from vllm.entrypoints.openai.models.serving import (
    OpenAIServingModels as OpenAIServingModels,
)

logger: Incomplete

def wrap_data_with_event(data: str, event: str): ...

class AnthropicServingMessages(OpenAIServingChat):
    stop_reason_map: Incomplete
    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        response_role: str,
        *,
        request_logger: RequestLogger | None,
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
        return_tokens_as_token_ids: bool = False,
        reasoning_parser: str = "",
        enable_auto_tools: bool = False,
        tool_parser: str | None = None,
        enable_prompt_tokens_details: bool = False,
        enable_force_include_usage: bool = False,
    ) -> None: ...
    async def create_messages(
        self, request: AnthropicMessagesRequest, raw_request: Request | None = None
    ) -> AsyncGenerator[str, None] | AnthropicMessagesResponse | ErrorResponse: ...
    def messages_full_converter(
        self, generator: ChatCompletionResponse
    ) -> AnthropicMessagesResponse: ...
    content_block_index: int
    block_type: str | None
    block_index: int | None
    block_signature: str | None
    signature_emitted: bool
    tool_use_id: str | None
    async def message_stream_converter(
        self, generator: AsyncGenerator[str, None]
    ) -> AsyncGenerator[str, None]: ...
    async def count_tokens(
        self, request: AnthropicCountTokensRequest, raw_request: Request | None = None
    ) -> AnthropicCountTokensResponse | ErrorResponse: ...
