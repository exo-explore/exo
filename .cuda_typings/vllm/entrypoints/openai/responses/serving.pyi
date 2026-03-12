import asyncio
from _typeshed import Incomplete
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator, Callable as Callable
from fastapi import Request as Request
from openai.types.responses import (
    ResponseOutputItem as ResponseOutputItem,
    ResponseStatus as ResponseStatus,
)
from openai.types.responses.tool import Tool as Tool
from typing import Final
from vllm import envs as envs
from vllm.config.utils import replace as replace
from vllm.engine.protocol import EngineClient as EngineClient
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam as ChatCompletionMessageParam,
    ChatTemplateContentFormatOption as ChatTemplateContentFormatOption,
)
from vllm.entrypoints.logger import RequestLogger as RequestLogger
from vllm.entrypoints.mcp.tool_server import ToolServer as ToolServer
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage as DeltaMessage,
    ErrorResponse as ErrorResponse,
    RequestResponseMetadata as RequestResponseMetadata,
)
from vllm.entrypoints.openai.engine.serving import (
    GenerationError as GenerationError,
    OpenAIServing as OpenAIServing,
)
from vllm.entrypoints.openai.models.serving import (
    OpenAIServingModels as OpenAIServingModels,
)
from vllm.entrypoints.openai.parser.harmony_utils import (
    get_developer_message as get_developer_message,
    get_stop_tokens_for_assistant_actions as get_stop_tokens_for_assistant_actions,
    get_system_message as get_system_message,
    get_user_message as get_user_message,
    has_custom_tools as has_custom_tools,
    render_for_completion as render_for_completion,
)
from vllm.entrypoints.openai.responses.context import (
    ConversationContext as ConversationContext,
    HarmonyContext as HarmonyContext,
    ParsableContext as ParsableContext,
    SimpleContext as SimpleContext,
    StreamingHarmonyContext as StreamingHarmonyContext,
)
from vllm.entrypoints.openai.responses.harmony import (
    construct_harmony_previous_input_messages as construct_harmony_previous_input_messages,
    harmony_to_response_output as harmony_to_response_output,
    parser_state_to_response_output as parser_state_to_response_output,
    response_input_to_harmony as response_input_to_harmony,
)
from vllm.entrypoints.openai.responses.protocol import (
    InputTokensDetails as InputTokensDetails,
    OutputTokensDetails as OutputTokensDetails,
    ResponseCompletedEvent as ResponseCompletedEvent,
    ResponseCreatedEvent as ResponseCreatedEvent,
    ResponseInProgressEvent as ResponseInProgressEvent,
    ResponseInputOutputMessage as ResponseInputOutputMessage,
    ResponseReasoningPartAddedEvent as ResponseReasoningPartAddedEvent,
    ResponseReasoningPartDoneEvent as ResponseReasoningPartDoneEvent,
    ResponseUsage as ResponseUsage,
    ResponsesRequest as ResponsesRequest,
    ResponsesResponse as ResponsesResponse,
    StreamingResponsesResponse as StreamingResponsesResponse,
)
from vllm.entrypoints.openai.responses.streaming_events import (
    StreamingState as StreamingState,
    emit_content_delta_events as emit_content_delta_events,
    emit_previous_item_done_events as emit_previous_item_done_events,
    emit_tool_action_events as emit_tool_action_events,
)
from vllm.entrypoints.openai.responses.utils import (
    construct_input_messages as construct_input_messages,
    construct_tool_dicts as construct_tool_dicts,
    extract_tool_types as extract_tool_types,
)
from vllm.entrypoints.utils import get_max_tokens as get_max_tokens
from vllm.exceptions import VLLMValidationError as VLLMValidationError
from vllm.inputs.data import (
    ProcessorInputs as ProcessorInputs,
    token_inputs as token_inputs,
)
from vllm.logger import init_logger as init_logger
from vllm.logprobs import SampleLogprobs as SampleLogprobs
from vllm.outputs import CompletionOutput as CompletionOutput
from vllm.parser import ParserManager as ParserManager
from vllm.sampling_params import (
    SamplingParams as SamplingParams,
    StructuredOutputsParams as StructuredOutputsParams,
)
from vllm.tokenizers import TokenizerLike as TokenizerLike
from vllm.utils import random_uuid as random_uuid

logger: Incomplete

class OpenAIServingResponses(OpenAIServing):
    chat_template: Incomplete
    chat_template_content_format: Final[Incomplete]
    enable_log_outputs: Incomplete
    parser: Incomplete
    enable_prompt_tokens_details: Incomplete
    enable_force_include_usage: Incomplete
    default_sampling_params: Incomplete
    override_max_tokens: Incomplete
    enable_store: Incomplete
    use_harmony: Incomplete
    tool_call_id_type: str
    enable_auto_tools: Incomplete
    response_store: dict[str, ResponsesResponse]
    response_store_lock: Incomplete
    msg_store: dict[str, list[ChatCompletionMessageParam]]
    event_store: dict[str, tuple[deque[StreamingResponsesResponse], asyncio.Event]]
    background_tasks: dict[str, asyncio.Task]
    tool_server: Incomplete
    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
        return_tokens_as_token_ids: bool = False,
        reasoning_parser: str = "",
        enable_auto_tools: bool = False,
        tool_parser: str | None = None,
        tool_server: ToolServer | None = None,
        enable_prompt_tokens_details: bool = False,
        enable_force_include_usage: bool = False,
        enable_log_outputs: bool = False,
    ) -> None: ...
    async def create_responses(
        self, request: ResponsesRequest, raw_request: Request | None = None
    ) -> (
        AsyncGenerator[StreamingResponsesResponse, None]
        | ResponsesResponse
        | ErrorResponse
    ): ...
    async def responses_full_generator(
        self,
        request: ResponsesRequest,
        sampling_params: SamplingParams,
        result_generator: AsyncIterator[ConversationContext],
        context: ConversationContext,
        model_name: str,
        tokenizer: TokenizerLike,
        request_metadata: RequestResponseMetadata,
        created_time: int | None = None,
    ) -> ErrorResponse | ResponsesResponse: ...
    async def responses_background_stream_generator(
        self, response_id: str, starting_after: int | None = None
    ) -> AsyncGenerator[StreamingResponsesResponse, None]: ...
    async def retrieve_responses(
        self, response_id: str, starting_after: int | None, stream: bool | None
    ) -> (
        ErrorResponse
        | ResponsesResponse
        | AsyncGenerator[StreamingResponsesResponse, None]
    ): ...
    async def cancel_responses(
        self, response_id: str
    ) -> ErrorResponse | ResponsesResponse: ...
    async def responses_stream_generator(
        self,
        request: ResponsesRequest,
        sampling_params: SamplingParams,
        result_generator: AsyncIterator[ConversationContext | None],
        context: ConversationContext,
        model_name: str,
        tokenizer: TokenizerLike,
        request_metadata: RequestResponseMetadata,
        created_time: int | None = None,
    ) -> AsyncGenerator[StreamingResponsesResponse, None]: ...
