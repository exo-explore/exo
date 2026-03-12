from _typeshed import Incomplete
from collections.abc import AsyncGenerator, Callable as Callable, Mapping
from dataclasses import dataclass, field
from fastapi import Request as Request
from http import HTTPStatus
from starlette.datastructures import Headers as Headers
from typing import ClassVar, Generic, Protocol, TypeAlias, TypeVar
from vllm.beam_search import (
    BeamSearchSequence as BeamSearchSequence,
    create_sort_beams_key_function as create_sort_beams_key_function,
)
from vllm.config import ModelConfig as ModelConfig
from vllm.engine.protocol import EngineClient as EngineClient
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam as ChatCompletionMessageParam,
    ChatTemplateContentFormatOption as ChatTemplateContentFormatOption,
    ConversationMessage as ConversationMessage,
)
from vllm.entrypoints.logger import RequestLogger as RequestLogger
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam as ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest as ChatCompletionRequest,
    ChatCompletionResponse as ChatCompletionResponse,
)
from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest as CompletionRequest,
    CompletionResponse as CompletionResponse,
)
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse as ErrorResponse,
    FunctionCall as FunctionCall,
    FunctionDefinition as FunctionDefinition,
    GenerationError as GenerationError,
)
from vllm.entrypoints.openai.models.serving import (
    OpenAIServingModels as OpenAIServingModels,
)
from vllm.entrypoints.openai.responses.context import (
    ConversationContext as ConversationContext,
    HarmonyContext as HarmonyContext,
    ParsableContext as ParsableContext,
    StreamingHarmonyContext as StreamingHarmonyContext,
)
from vllm.entrypoints.openai.responses.protocol import (
    ResponseInputOutputItem as ResponseInputOutputItem,
    ResponsesRequest as ResponsesRequest,
)
from vllm.entrypoints.openai.responses.utils import (
    construct_input_messages as construct_input_messages,
)
from vllm.entrypoints.openai.speech_to_text.protocol import (
    TranscriptionRequest as TranscriptionRequest,
    TranscriptionResponse as TranscriptionResponse,
    TranslationRequest as TranslationRequest,
)
from vllm.entrypoints.pooling.pooling.protocol import (
    IOProcessorRequest as IOProcessorRequest,
    PoolingChatRequest as PoolingChatRequest,
    PoolingCompletionRequest as PoolingCompletionRequest,
    PoolingResponse as PoolingResponse,
)
from vllm.entrypoints.pooling.score.protocol import (
    RerankRequest as RerankRequest,
    ScoreDataRequest as ScoreDataRequest,
    ScoreQueriesDocumentsRequest as ScoreQueriesDocumentsRequest,
    ScoreRequest as ScoreRequest,
    ScoreResponse as ScoreResponse,
    ScoreTextRequest as ScoreTextRequest,
)
from vllm.entrypoints.serve.disagg.protocol import (
    GenerateRequest as GenerateRequest,
    GenerateResponse as GenerateResponse,
)
from vllm.entrypoints.serve.tokenize.protocol import (
    DetokenizeRequest as DetokenizeRequest,
    TokenizeChatRequest as TokenizeChatRequest,
    TokenizeCompletionRequest as TokenizeCompletionRequest,
    TokenizeResponse as TokenizeResponse,
)
from vllm.entrypoints.utils import (
    create_error_response as create_error_response,
    get_max_tokens as get_max_tokens,
)
from vllm.exceptions import VLLMValidationError as VLLMValidationError
from vllm.inputs.data import (
    ProcessorInputs as ProcessorInputs,
    PromptType as PromptType,
    SingletonPrompt as SingletonPrompt,
    TokensPrompt as TokensPrompt,
    token_inputs as token_inputs,
)
from vllm.logger import init_logger as init_logger
from vllm.logprobs import Logprob as Logprob, PromptLogprobs as PromptLogprobs
from vllm.lora.request import LoRARequest as LoRARequest
from vllm.outputs import (
    CompletionOutput as CompletionOutput,
    PoolingRequestOutput as PoolingRequestOutput,
    RequestOutput as RequestOutput,
)
from vllm.pooling_params import PoolingParams as PoolingParams
from vllm.renderers import (
    ChatParams as ChatParams,
    TokenizeParams as TokenizeParams,
    merge_kwargs as merge_kwargs,
)
from vllm.renderers.inputs.preprocess import (
    extract_prompt_components as extract_prompt_components,
    extract_prompt_len as extract_prompt_len,
    parse_model_prompt as parse_model_prompt,
    prompt_to_seq as prompt_to_seq,
)
from vllm.sampling_params import (
    BeamSearchParams as BeamSearchParams,
    SamplingParams as SamplingParams,
)
from vllm.tokenizers import TokenizerLike as TokenizerLike
from vllm.tool_parsers import ToolParser as ToolParser
from vllm.tracing import (
    contains_trace_headers as contains_trace_headers,
    extract_trace_headers as extract_trace_headers,
    log_tracing_disabled_warning as log_tracing_disabled_warning,
)
from vllm.utils import random_uuid as random_uuid
from vllm.utils.async_utils import (
    collect_from_async_generator as collect_from_async_generator,
    merge_async_iterators as merge_async_iterators,
)
from vllm.utils.mistral import is_mistral_tokenizer as is_mistral_tokenizer

logger: Incomplete

class RendererRequest(Protocol):
    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams: ...

class RendererChatRequest(RendererRequest, Protocol):
    def build_chat_params(
        self,
        default_template: str | None,
        default_template_content_format: ChatTemplateContentFormatOption,
    ) -> ChatParams: ...

CompletionLikeRequest: TypeAlias = (
    CompletionRequest
    | TokenizeCompletionRequest
    | DetokenizeRequest
    | RerankRequest
    | ScoreRequest
    | PoolingCompletionRequest
)
ChatLikeRequest: TypeAlias = (
    ChatCompletionRequest | TokenizeChatRequest | PoolingChatRequest
)
SpeechToTextRequest: TypeAlias = TranscriptionRequest | TranslationRequest
AnyRequest: TypeAlias = (
    CompletionLikeRequest
    | ChatLikeRequest
    | SpeechToTextRequest
    | ResponsesRequest
    | IOProcessorRequest
    | GenerateRequest
)
AnyResponse: TypeAlias = (
    CompletionResponse
    | ChatCompletionResponse
    | TranscriptionResponse
    | TokenizeResponse
    | PoolingResponse
    | ScoreResponse
    | GenerateResponse
)
RequestT = TypeVar("RequestT", bound=AnyRequest)

@dataclass(kw_only=True)
class ServeContext(Generic[RequestT]):
    request: RequestT
    raw_request: Request | None = ...
    model_name: str
    request_id: str
    created_time: int = field(default_factory=Incomplete)
    lora_request: LoRARequest | None = ...
    engine_prompts: list[ProcessorInputs] | None = ...
    result_generator: (
        AsyncGenerator[tuple[int, PoolingRequestOutput], None] | None
    ) = ...
    final_res_batch: list[PoolingRequestOutput] = field(default_factory=list)
    model_config = ...

class OpenAIServing:
    request_id_prefix: ClassVar[str]
    engine_client: Incomplete
    models: Incomplete
    request_logger: Incomplete
    return_tokens_as_token_ids: Incomplete
    model_config: Incomplete
    renderer: Incomplete
    io_processor: Incomplete
    input_processor: Incomplete
    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        return_tokens_as_token_ids: bool = False,
    ) -> None: ...
    async def beam_search(
        self,
        prompt: ProcessorInputs,
        request_id: str,
        params: BeamSearchParams,
        lora_request: LoRARequest | None = None,
        trace_headers: Mapping[str, str] | None = None,
    ) -> AsyncGenerator[RequestOutput, None]: ...
    async def handle(self, ctx: ServeContext) -> AnyResponse | ErrorResponse: ...
    @staticmethod
    def create_error_response(
        message: str | Exception,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = ...,
        param: str | None = None,
    ) -> ErrorResponse: ...
    def create_streaming_error_response(
        self,
        message: str | Exception,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = ...,
        param: str | None = None,
    ) -> str: ...

def clamp_prompt_logprobs(
    prompt_logprobs: PromptLogprobs | None,
) -> PromptLogprobs | None: ...
