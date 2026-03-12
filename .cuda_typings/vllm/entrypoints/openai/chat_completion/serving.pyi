from _typeshed import Incomplete
from collections.abc import AsyncGenerator, AsyncIterator
from fastapi import Request as Request
from typing import Any, Final
from vllm.engine.protocol import EngineClient as EngineClient
from vllm.entrypoints.chat_utils import (
    ChatTemplateContentFormatOption as ChatTemplateContentFormatOption,
    ConversationMessage as ConversationMessage,
    get_history_tool_calls_cnt as get_history_tool_calls_cnt,
    make_tool_call_id as make_tool_call_id,
)
from vllm.entrypoints.logger import RequestLogger as RequestLogger
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionLogProb as ChatCompletionLogProb,
    ChatCompletionLogProbs as ChatCompletionLogProbs,
    ChatCompletionLogProbsContent as ChatCompletionLogProbsContent,
    ChatCompletionNamedToolChoiceParam as ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest as ChatCompletionRequest,
    ChatCompletionResponse as ChatCompletionResponse,
    ChatCompletionResponseChoice as ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice as ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse as ChatCompletionStreamResponse,
    ChatMessage as ChatMessage,
)
from vllm.entrypoints.openai.chat_completion.stream_harmony import (
    TokenState as TokenState,
    extract_harmony_streaming_delta as extract_harmony_streaming_delta,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall as DeltaFunctionCall,
    DeltaMessage as DeltaMessage,
    DeltaToolCall as DeltaToolCall,
    ErrorResponse as ErrorResponse,
    FunctionCall as FunctionCall,
    PromptTokenUsageInfo as PromptTokenUsageInfo,
    RequestResponseMetadata as RequestResponseMetadata,
    ToolCall as ToolCall,
    UsageInfo as UsageInfo,
)
from vllm.entrypoints.openai.engine.serving import (
    GenerationError as GenerationError,
    OpenAIServing as OpenAIServing,
    clamp_prompt_logprobs as clamp_prompt_logprobs,
)
from vllm.entrypoints.openai.models.serving import (
    OpenAIServingModels as OpenAIServingModels,
)
from vllm.entrypoints.openai.parser.harmony_utils import (
    get_developer_message as get_developer_message,
    get_stop_tokens_for_assistant_actions as get_stop_tokens_for_assistant_actions,
    get_streamable_parser_for_assistant as get_streamable_parser_for_assistant,
    get_system_message as get_system_message,
    parse_chat_inputs_to_harmony_messages as parse_chat_inputs_to_harmony_messages,
    parse_chat_output as parse_chat_output,
    render_for_completion as render_for_completion,
)
from vllm.entrypoints.openai.utils import (
    maybe_filter_parallel_tool_calls as maybe_filter_parallel_tool_calls,
)
from vllm.entrypoints.utils import (
    get_max_tokens as get_max_tokens,
    should_include_usage as should_include_usage,
)
from vllm.inputs.data import (
    ProcessorInputs as ProcessorInputs,
    TokensPrompt as TokensPrompt,
)
from vllm.logger import init_logger as init_logger
from vllm.logprobs import Logprob as Logprob
from vllm.outputs import (
    CompletionOutput as CompletionOutput,
    RequestOutput as RequestOutput,
)
from vllm.parser import ParserManager as ParserManager
from vllm.reasoning import ReasoningParser as ReasoningParser
from vllm.renderers import ChatParams as ChatParams
from vllm.sampling_params import (
    BeamSearchParams as BeamSearchParams,
    SamplingParams as SamplingParams,
)
from vllm.tokenizers import TokenizerLike as TokenizerLike
from vllm.tool_parsers import ToolParser as ToolParser
from vllm.tool_parsers.mistral_tool_parser import MistralToolCall as MistralToolCall
from vllm.tool_parsers.utils import partial_json_loads as partial_json_loads
from vllm.utils.collection_utils import as_list as as_list
from vllm.utils.mistral import is_mistral_tokenizer as is_mistral_tokenizer

logger: Incomplete

class OpenAIServingChat(OpenAIServing):
    response_role: Incomplete
    chat_template: Incomplete
    chat_template_content_format: Final[Incomplete]
    trust_request_chat_template: Incomplete
    default_chat_template_kwargs: Incomplete
    enable_log_outputs: Incomplete
    enable_log_deltas: Incomplete
    reasoning_parser_cls: Incomplete
    enable_auto_tools: bool
    tool_parser: Incomplete
    exclude_tools_when_tool_choice_none: Incomplete
    enable_prompt_tokens_details: Incomplete
    enable_force_include_usage: Incomplete
    default_sampling_params: Incomplete
    override_max_tokens: Incomplete
    use_harmony: Incomplete
    tool_call_id_type: str
    supports_browsing: bool
    browser_tool: Incomplete
    supports_code_interpreter: bool
    python_tool: Incomplete
    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        response_role: str,
        *,
        request_logger: RequestLogger | None,
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
        trust_request_chat_template: bool = False,
        return_tokens_as_token_ids: bool = False,
        reasoning_parser: str = "",
        enable_auto_tools: bool = False,
        exclude_tools_when_tool_choice_none: bool = False,
        tool_parser: str | None = None,
        enable_prompt_tokens_details: bool = False,
        enable_force_include_usage: bool = False,
        enable_log_outputs: bool = False,
        enable_log_deltas: bool = True,
        default_chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None: ...
    def warmup(self) -> None: ...
    async def render_chat_request(
        self, request: ChatCompletionRequest
    ) -> tuple[list[ConversationMessage], list[ProcessorInputs]] | ErrorResponse: ...
    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request | None = None
    ) -> AsyncGenerator[str, None] | ChatCompletionResponse | ErrorResponse: ...
    def get_chat_request_role(self, request: ChatCompletionRequest) -> str: ...
    def extract_tool_call_required_streaming(
        self,
        previous_text: str,
        current_text: str | None,
        delta_text: str,
        function_name_returned: bool,
        tool_call_idx: int | None = None,
    ) -> tuple[DeltaMessage | None, bool]: ...
    async def chat_completion_stream_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation: list[ConversationMessage],
        tokenizer: TokenizerLike,
        request_metadata: RequestResponseMetadata,
        reasoning_parser: ReasoningParser | None = None,
    ) -> AsyncGenerator[str, None]: ...
    async def chat_completion_full_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation: list[ConversationMessage],
        tokenizer: TokenizerLike,
        request_metadata: RequestResponseMetadata,
        reasoning_parser: ReasoningParser | None = None,
    ) -> ErrorResponse | ChatCompletionResponse: ...
