from _typeshed import Incomplete
from pydantic import BaseModel
from typing import Any, ClassVar, Literal, TypeAlias
from vllm.entrypoints.chat_utils import make_tool_call_id as make_tool_call_id
from vllm.logger import init_logger as init_logger
from vllm.sampling_params import SamplingParams as SamplingParams
from vllm.utils import random_uuid as random_uuid
from vllm.utils.import_utils import resolve_obj_by_qualname as resolve_obj_by_qualname

logger: Incomplete

class OpenAIBaseModel(BaseModel):
    model_config: Incomplete
    field_names: ClassVar[set[str] | None]
    @classmethod
    def __log_extra_fields__(cls, data, handler): ...

class ErrorInfo(OpenAIBaseModel):
    message: str
    type: str
    param: str | None
    code: int

class ErrorResponse(OpenAIBaseModel):
    error: ErrorInfo

class ModelPermission(OpenAIBaseModel):
    id: str
    object: str
    created: int
    allow_create_engine: bool
    allow_sampling: bool
    allow_logprobs: bool
    allow_search_indices: bool
    allow_view: bool
    allow_fine_tuning: bool
    organization: str
    group: str | None
    is_blocking: bool

class ModelCard(OpenAIBaseModel):
    id: str
    object: str
    created: int
    owned_by: str
    root: str | None
    parent: str | None
    max_model_len: int | None
    permission: list[ModelPermission]

class ModelList(OpenAIBaseModel):
    object: str
    data: list[ModelCard]

class PromptTokenUsageInfo(OpenAIBaseModel):
    cached_tokens: int | None

class UsageInfo(OpenAIBaseModel):
    prompt_tokens: int
    total_tokens: int
    completion_tokens: int | None
    prompt_tokens_details: PromptTokenUsageInfo | None

class RequestResponseMetadata(BaseModel):
    request_id: str
    final_usage_info: UsageInfo | None

class JsonSchemaResponseFormat(OpenAIBaseModel):
    name: str
    description: str | None
    json_schema: dict[str, Any] | None
    strict: bool | None

class LegacyStructuralTag(OpenAIBaseModel):
    begin: str
    structural_tag_schema: dict[str, Any] | None
    end: str

class LegacyStructuralTagResponseFormat(OpenAIBaseModel):
    type: Literal["structural_tag"]
    structures: list[LegacyStructuralTag]
    triggers: list[str]

class StructuralTagResponseFormat(OpenAIBaseModel):
    type: Literal["structural_tag"]
    format: Any

AnyStructuralTagResponseFormat: TypeAlias = (
    LegacyStructuralTagResponseFormat | StructuralTagResponseFormat
)

class ResponseFormat(OpenAIBaseModel):
    type: Literal["text", "json_object", "json_schema"]
    json_schema: JsonSchemaResponseFormat | None

AnyResponseFormat: TypeAlias = (
    ResponseFormat | StructuralTagResponseFormat | LegacyStructuralTagResponseFormat
)

class StreamOptions(OpenAIBaseModel):
    include_usage: bool | None
    continuous_usage_stats: bool | None

class FunctionDefinition(OpenAIBaseModel):
    name: str
    description: str | None
    parameters: dict[str, Any] | None

class LogitsProcessorConstructor(BaseModel):
    qualname: str
    args: list[Any] | None
    kwargs: dict[str, Any] | None
    model_config: Incomplete

LogitsProcessors = list[str | LogitsProcessorConstructor]

def get_logits_processors(
    processors: LogitsProcessors | None, pattern: str | None
) -> list[Any] | None: ...

class FunctionCall(OpenAIBaseModel):
    id: str | None
    name: str
    arguments: str

class ToolCall(OpenAIBaseModel):
    id: str
    type: Literal["function"]
    function: FunctionCall

class DeltaFunctionCall(BaseModel):
    name: str | None
    arguments: str | None

class DeltaToolCall(OpenAIBaseModel):
    id: str | None
    type: Literal["function"] | None
    index: int
    function: DeltaFunctionCall | None

class ExtractedToolCallInformation(BaseModel):
    tools_called: bool
    tool_calls: list[ToolCall]
    content: str | None

class DeltaMessage(OpenAIBaseModel):
    role: str | None
    content: str | None
    reasoning: str | None
    tool_calls: list[DeltaToolCall]

class GenerationError(Exception):
    status_code: Incomplete
    def __init__(self, message: str = "Internal server error") -> None: ...

class GenerateRequest(BaseModel):
    request_id: str
    token_ids: list[int]
    features: str | None
    sampling_params: SamplingParams
    model: str | None
    stream: bool | None
    stream_options: StreamOptions | None
    cache_salt: str | None
    priority: int
    kv_transfer_params: dict[str, Any] | None
