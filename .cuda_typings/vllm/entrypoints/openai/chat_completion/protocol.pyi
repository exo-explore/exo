from _typeshed import Incomplete
from openai.types.chat.chat_completion_audio import (
    ChatCompletionAudio as OpenAIChatCompletionAudio,
)
from openai.types.chat.chat_completion_message import Annotation as OpenAIAnnotation
from typing import Annotated, Any, ClassVar, Literal
from vllm.config import ModelConfig as ModelConfig
from vllm.config.utils import replace as replace
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam as ChatCompletionMessageParam,
    ChatTemplateContentFormatOption as ChatTemplateContentFormatOption,
)
from vllm.entrypoints.openai.engine.protocol import (
    AnyResponseFormat as AnyResponseFormat,
    DeltaMessage as DeltaMessage,
    FunctionCall as FunctionCall,
    FunctionDefinition as FunctionDefinition,
    LegacyStructuralTagResponseFormat as LegacyStructuralTagResponseFormat,
    OpenAIBaseModel as OpenAIBaseModel,
    StreamOptions as StreamOptions,
    StructuralTagResponseFormat as StructuralTagResponseFormat,
    ToolCall as ToolCall,
    UsageInfo as UsageInfo,
)
from vllm.exceptions import VLLMValidationError as VLLMValidationError
from vllm.logger import init_logger as init_logger
from vllm.logprobs import Logprob as Logprob
from vllm.renderers import (
    ChatParams as ChatParams,
    TokenizeParams as TokenizeParams,
    merge_kwargs as merge_kwargs,
)
from vllm.sampling_params import (
    BeamSearchParams as BeamSearchParams,
    RepetitionDetectionParams as RepetitionDetectionParams,
    RequestOutputKind as RequestOutputKind,
    SamplingParams as SamplingParams,
    StructuredOutputsParams as StructuredOutputsParams,
)
from vllm.utils import random_uuid as random_uuid

logger: Incomplete

class ChatMessage(OpenAIBaseModel):
    role: str
    content: str | None
    refusal: str | None
    annotations: OpenAIAnnotation | None
    audio: OpenAIChatCompletionAudio | None
    function_call: FunctionCall | None
    tool_calls: list[ToolCall]
    reasoning: str | None

class ChatCompletionLogProb(OpenAIBaseModel):
    token: str
    logprob: float
    bytes: list[int] | None

class ChatCompletionLogProbsContent(ChatCompletionLogProb):
    field_names: ClassVar[set[str] | None]
    top_logprobs: list[ChatCompletionLogProb]

class ChatCompletionLogProbs(OpenAIBaseModel):
    content: list[ChatCompletionLogProbsContent] | None

class ChatCompletionResponseChoice(OpenAIBaseModel):
    index: int
    message: ChatMessage
    logprobs: ChatCompletionLogProbs | None
    finish_reason: str | None
    stop_reason: int | str | None
    token_ids: list[int] | None

class ChatCompletionResponse(OpenAIBaseModel):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: list[ChatCompletionResponseChoice]
    service_tier: Literal["auto", "default", "flex", "scale", "priority"] | None
    system_fingerprint: str | None
    usage: UsageInfo
    prompt_logprobs: list[dict[int, Logprob] | None] | None
    prompt_token_ids: list[int] | None
    kv_transfer_params: dict[str, Any] | None

class ChatCompletionResponseStreamChoice(OpenAIBaseModel):
    index: int
    delta: DeltaMessage
    logprobs: ChatCompletionLogProbs | None
    finish_reason: str | None
    stop_reason: int | str | None
    token_ids: list[int] | None

class ChatCompletionStreamResponse(OpenAIBaseModel):
    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    choices: list[ChatCompletionResponseStreamChoice]
    usage: UsageInfo | None
    prompt_token_ids: list[int] | None

class ChatCompletionToolsParam(OpenAIBaseModel):
    type: Literal["function"]
    function: FunctionDefinition

class ChatCompletionNamedFunction(OpenAIBaseModel):
    name: str

class ChatCompletionNamedToolChoiceParam(OpenAIBaseModel):
    function: ChatCompletionNamedFunction
    type: Literal["function"]

class ChatCompletionRequest(OpenAIBaseModel):
    messages: list[ChatCompletionMessageParam]
    model: str | None
    frequency_penalty: float | None
    logit_bias: dict[str, float] | None
    logprobs: bool | None
    top_logprobs: int | None
    max_tokens: int | None
    max_completion_tokens: int | None
    n: int | None
    presence_penalty: float | None
    response_format: AnyResponseFormat | None
    seed: int | None
    stop: str | list[str] | None
    stream: bool | None
    stream_options: StreamOptions | None
    temperature: float | None
    top_p: float | None
    tools: list[ChatCompletionToolsParam] | None
    tool_choice: (
        Literal["none"]
        | Literal["auto"]
        | Literal["required"]
        | ChatCompletionNamedToolChoiceParam
        | None
    )
    reasoning_effort: Literal["low", "medium", "high"] | None
    include_reasoning: bool
    parallel_tool_calls: bool | None
    user: str | None
    use_beam_search: bool
    top_k: int | None
    min_p: float | None
    repetition_penalty: float | None
    length_penalty: float
    stop_token_ids: list[int] | None
    include_stop_str_in_output: bool
    ignore_eos: bool
    min_tokens: int
    skip_special_tokens: bool
    spaces_between_special_tokens: bool
    truncate_prompt_tokens: Annotated[int, None] | None
    prompt_logprobs: int | None
    allowed_token_ids: list[int] | None
    bad_words: list[str]
    echo: bool
    add_generation_prompt: bool
    continue_final_message: bool
    add_special_tokens: bool
    documents: list[dict[str, str]] | None
    chat_template: str | None
    chat_template_kwargs: dict[str, Any] | None
    media_io_kwargs: dict[str, dict[str, Any]] | None
    mm_processor_kwargs: dict[str, Any] | None
    structured_outputs: StructuredOutputsParams | None
    priority: int
    request_id: str
    return_tokens_as_token_ids: bool | None
    return_token_ids: bool | None
    cache_salt: str | None
    kv_transfer_params: dict[str, Any] | None
    vllm_xargs: dict[str, str | int | float | list[str | int | float]] | None
    repetition_detection: RepetitionDetectionParams | None
    def build_chat_params(
        self,
        default_template: str | None,
        default_template_content_format: ChatTemplateContentFormatOption,
    ) -> ChatParams: ...
    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams: ...
    def to_beam_search_params(
        self, max_tokens: int, default_sampling_params: dict
    ) -> BeamSearchParams: ...
    def to_sampling_params(
        self, max_tokens: int, default_sampling_params: dict
    ) -> SamplingParams: ...
    @classmethod
    def validate_response_format(cls, data): ...
    @classmethod
    def validate_stream_options(cls, data): ...
    @classmethod
    def check_logprobs(cls, data): ...
    @classmethod
    def check_structured_outputs_count(cls, data): ...
    @classmethod
    def check_tool_usage(cls, data): ...
    @classmethod
    def check_generation_prompt(cls, data): ...
    @classmethod
    def check_cache_salt_support(cls, data): ...
    @classmethod
    def check_system_message_content_type(cls, data): ...
