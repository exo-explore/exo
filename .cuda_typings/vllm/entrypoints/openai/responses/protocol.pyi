from _typeshed import Incomplete
from openai.types.responses import (
    ResponseCodeInterpreterCallCodeDeltaEvent,
    ResponseCodeInterpreterCallCodeDoneEvent,
    ResponseCodeInterpreterCallCompletedEvent,
    ResponseCodeInterpreterCallInProgressEvent,
    ResponseCodeInterpreterCallInterpretingEvent,
    ResponseCompletedEvent as OpenAIResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent as OpenAIResponseCreatedEvent,
    ResponseFormatTextConfig as ResponseTextConfig,
    ResponseInProgressEvent as OpenAIResponseInProgressEvent,
    ResponseInputItemParam,
    ResponseMcpCallArgumentsDeltaEvent,
    ResponseMcpCallArgumentsDoneEvent,
    ResponseMcpCallCompletedEvent,
    ResponseMcpCallInProgressEvent,
    ResponseOutputItem,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponsePrompt as ResponsePrompt,
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent,
    ResponseStatus as ResponseStatus,
    ResponseWebSearchCallCompletedEvent,
    ResponseWebSearchCallInProgressEvent,
    ResponseWebSearchCallSearchingEvent,
)
from openai.types.responses.response import IncompleteDetails, ToolChoice as ToolChoice
from openai.types.responses.response_reasoning_item import (
    Content as ResponseReasoningTextContent,
)
from openai.types.responses.tool import Tool as Tool
from openai.types.shared import Metadata as Metadata, Reasoning as Reasoning
from openai_harmony import Message as OpenAIHarmonyMessage
from typing import Any, Literal, TypeAlias
from vllm.config import ModelConfig as ModelConfig
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam as ChatCompletionMessageParam,
    ChatTemplateContentFormatOption as ChatTemplateContentFormatOption,
)
from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel as OpenAIBaseModel
from vllm.exceptions import VLLMValidationError as VLLMValidationError
from vllm.logger import init_logger as init_logger
from vllm.renderers import (
    ChatParams as ChatParams,
    TokenizeParams as TokenizeParams,
    merge_kwargs as merge_kwargs,
)
from vllm.sampling_params import (
    RequestOutputKind as RequestOutputKind,
    SamplingParams as SamplingParams,
    StructuredOutputsParams as StructuredOutputsParams,
)
from vllm.utils import random_uuid as random_uuid

logger: Incomplete

class InputTokensDetails(OpenAIBaseModel):
    cached_tokens: int
    input_tokens_per_turn: list[int]
    cached_tokens_per_turn: list[int]

class OutputTokensDetails(OpenAIBaseModel):
    reasoning_tokens: int
    tool_output_tokens: int
    output_tokens_per_turn: list[int]
    tool_output_tokens_per_turn: list[int]

class ResponseUsage(OpenAIBaseModel):
    input_tokens: int
    input_tokens_details: InputTokensDetails
    output_tokens: int
    output_tokens_details: OutputTokensDetails
    total_tokens: int

def serialize_message(msg): ...
def serialize_messages(msgs): ...

class ResponseRawMessageAndToken(OpenAIBaseModel):
    message: str
    tokens: list[int]
    type: Literal["raw_message_tokens"]

ResponseInputOutputMessage: TypeAlias = (
    list[ChatCompletionMessageParam] | list[ResponseRawMessageAndToken]
)
ResponseInputOutputItem: TypeAlias = ResponseInputItemParam | ResponseOutputItem

class ResponsesRequest(OpenAIBaseModel):
    background: bool | None
    include: (
        list[
            Literal[
                "code_interpreter_call.outputs",
                "computer_call_output.output.image_url",
                "file_search_call.results",
                "message.input_image.image_url",
                "message.output_text.logprobs",
                "reasoning.encrypted_content",
            ]
        ]
        | None
    )
    input: str | list[ResponseInputOutputItem]
    instructions: str | None
    max_output_tokens: int | None
    max_tool_calls: int | None
    metadata: Metadata | None
    model: str | None
    logit_bias: dict[str, float] | None
    parallel_tool_calls: bool | None
    previous_response_id: str | None
    prompt: ResponsePrompt | None
    reasoning: Reasoning | None
    service_tier: Literal["auto", "default", "flex", "scale", "priority"]
    store: bool | None
    stream: bool | None
    temperature: float | None
    text: ResponseTextConfig | None
    tool_choice: ToolChoice
    tools: list[Tool]
    top_logprobs: int | None
    top_p: float | None
    top_k: int | None
    truncation: Literal["auto", "disabled"] | None
    user: str | None
    skip_special_tokens: bool
    include_stop_str_in_output: bool
    prompt_cache_key: str | None
    request_id: str
    media_io_kwargs: dict[str, dict[str, Any]] | None
    mm_processor_kwargs: dict[str, Any] | None
    priority: int
    cache_salt: str | None
    enable_response_messages: bool
    previous_input_messages: list[OpenAIHarmonyMessage | dict] | None
    structured_outputs: StructuredOutputsParams | None
    repetition_penalty: float | None
    seed: int | None
    stop: str | list[str] | None
    ignore_eos: bool
    vllm_xargs: dict[str, str | int | float | list[str | int | float]] | None
    def build_chat_params(
        self,
        default_template: str | None,
        default_template_content_format: ChatTemplateContentFormatOption,
    ) -> ChatParams: ...
    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams: ...
    def to_sampling_params(
        self, default_max_tokens: int, default_sampling_params: dict | None = None
    ) -> SamplingParams: ...
    def is_include_output_logprobs(self) -> bool: ...
    @classmethod
    def validate_background(cls, data): ...
    @classmethod
    def validate_prompt(cls, data): ...
    @classmethod
    def check_cache_salt_support(cls, data): ...
    @classmethod
    def function_call_parsing(cls, data): ...

class ResponsesResponse(OpenAIBaseModel):
    id: str
    created_at: int
    incomplete_details: IncompleteDetails | None
    instructions: str | None
    metadata: Metadata | None
    model: str
    object: Literal["response"]
    output: list[ResponseOutputItem]
    parallel_tool_calls: bool
    temperature: float
    tool_choice: ToolChoice
    tools: list[Tool]
    top_p: float
    background: bool
    max_output_tokens: int
    max_tool_calls: int | None
    previous_response_id: str | None
    prompt: ResponsePrompt | None
    reasoning: Reasoning | None
    service_tier: Literal["auto", "default", "flex", "scale", "priority"]
    status: ResponseStatus
    text: ResponseTextConfig | None
    top_logprobs: int | None
    truncation: Literal["auto", "disabled"]
    usage: ResponseUsage | None
    user: str | None
    input_messages: ResponseInputOutputMessage | None
    output_messages: ResponseInputOutputMessage | None
    def serialize_output_messages(self, msgs, _info): ...
    def serialize_input_messages(self, msgs, _info): ...
    @classmethod
    def from_request(
        cls,
        request: ResponsesRequest,
        sampling_params: SamplingParams,
        model_name: str,
        created_time: int,
        output: list[ResponseOutputItem],
        status: ResponseStatus,
        usage: ResponseUsage | None = None,
        input_messages: ResponseInputOutputMessage | None = None,
        output_messages: ResponseInputOutputMessage | None = None,
    ) -> ResponsesResponse: ...

class ResponseReasoningPartDoneEvent(OpenAIBaseModel):
    content_index: int
    item_id: str
    output_index: int
    part: ResponseReasoningTextContent
    sequence_number: int
    type: Literal["response.reasoning_part.done"]

class ResponseReasoningPartAddedEvent(OpenAIBaseModel):
    content_index: int
    item_id: str
    output_index: int
    part: ResponseReasoningTextContent
    sequence_number: int
    type: Literal["response.reasoning_part.added"]

class ResponseCompletedEvent(OpenAIResponseCompletedEvent):
    response: ResponsesResponse

class ResponseCreatedEvent(OpenAIResponseCreatedEvent):
    response: ResponsesResponse

class ResponseInProgressEvent(OpenAIResponseInProgressEvent):
    response: ResponsesResponse

StreamingResponsesResponse: TypeAlias = (
    ResponseCreatedEvent
    | ResponseInProgressEvent
    | ResponseCompletedEvent
    | ResponseOutputItemAddedEvent
    | ResponseOutputItemDoneEvent
    | ResponseContentPartAddedEvent
    | ResponseContentPartDoneEvent
    | ResponseReasoningTextDeltaEvent
    | ResponseReasoningTextDoneEvent
    | ResponseReasoningPartAddedEvent
    | ResponseReasoningPartDoneEvent
    | ResponseCodeInterpreterCallInProgressEvent
    | ResponseCodeInterpreterCallCodeDeltaEvent
    | ResponseWebSearchCallInProgressEvent
    | ResponseWebSearchCallSearchingEvent
    | ResponseWebSearchCallCompletedEvent
    | ResponseCodeInterpreterCallCodeDoneEvent
    | ResponseCodeInterpreterCallInterpretingEvent
    | ResponseCodeInterpreterCallCompletedEvent
    | ResponseMcpCallArgumentsDeltaEvent
    | ResponseMcpCallArgumentsDoneEvent
    | ResponseMcpCallInProgressEvent
    | ResponseMcpCallCompletedEvent
)
