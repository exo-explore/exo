import time
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_core import PydanticUseDefault

from exo.shared.types.common import CommandId
from exo.shared.types.memory import Memory
from exo.shared.types.models import ModelId, ModelMetadata
from exo.shared.types.worker.instances import Instance, InstanceId, InstanceMeta
from exo.shared.types.worker.shards import Sharding

FinishReason = Literal[
    "stop", "length", "tool_calls", "content_filter", "function_call", "error"
]


class ErrorInfo(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: int


class ErrorResponse(BaseModel):
    error: ErrorInfo


class ModelListModel(BaseModel):
    id: str  # Full HuggingFace model_id (e.g., "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "exo"
    # openwebui fields
    name: str = Field(default="")
    description: str = Field(default="")
    context_length: int = Field(default=0)
    tags: list[str] = Field(default=[])
    storage_size_megabytes: int = Field(default=0)
    supports_tensor: bool = Field(default=False)
    # Grouped model fields (new)
    base_model: str = Field(default="")  # Base model id (e.g., "llama-3.1-8b")
    base_model_name: str = Field(default="")  # Display name (e.g., "Llama 3.1 8B")
    quantization: str = Field(default="")  # Quantization type (e.g., "4bit", "8bit")
    architecture: str = Field(default="")  # HuggingFace model_type (e.g., "llama")
    # UI display fields
    tagline: str = Field(default="")  # Short description (max 60 chars)
    capabilities: list[str] = Field(default=[])  # e.g., ["text", "thinking", "code"]
    family: str = Field(default="")  # Model family (e.g., "llama", "qwen")


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelListModel]


class ChatCompletionMessageText(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ChatCompletionMessage(BaseModel):
    role: Literal["system", "user", "assistant", "developer", "tool", "function"]
    content: (
        str | ChatCompletionMessageText | list[ChatCompletionMessageText] | None
    ) = None
    thinking: str | None = None  # Added for GPT-OSS harmony format support
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    function_call: dict[str, Any] | None = None


class BenchChatCompletionMessage(ChatCompletionMessage):
    pass


class TopLogprobItem(BaseModel):
    token: str
    logprob: float
    bytes: list[int] | None = None


class LogprobsContentItem(BaseModel):
    token: str
    logprob: float
    bytes: list[int] | None = None
    top_logprobs: list[TopLogprobItem]


class Logprobs(BaseModel):
    content: list[LogprobsContentItem] | None = None


class PromptTokensDetails(BaseModel):
    cached_tokens: int = 0
    audio_tokens: int = 0


class CompletionTokensDetails(BaseModel):
    reasoning_tokens: int = 0
    audio_tokens: int = 0
    accepted_prediction_tokens: int = 0
    rejected_prediction_tokens: int = 0


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: PromptTokensDetails | None = None
    completion_tokens_details: CompletionTokensDetails | None = None


class StreamingChoiceResponse(BaseModel):
    index: int
    delta: ChatCompletionMessage
    logprobs: Logprobs | None = None
    finish_reason: FinishReason | None = None
    usage: Usage | None = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    logprobs: Logprobs | None = None
    finish_reason: FinishReason | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice | StreamingChoiceResponse]
    usage: Usage | None = None
    service_tier: str | None = None


class GenerationStats(BaseModel):
    prompt_tps: float
    generation_tps: float
    prompt_tokens: int
    generation_tokens: int
    peak_memory_usage: Memory


class BenchChatCompletionResponse(ChatCompletionResponse):
    generation_stats: GenerationStats | None = None


class ChatCompletionTaskParams(BaseModel):
    model: str
    frequency_penalty: float | None = None
    messages: list[ChatCompletionMessage]
    logit_bias: dict[str, int] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    max_tokens: int | None = None
    n: int | None = None
    presence_penalty: float | None = None
    response_format: dict[str, Any] | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    user: str | None = None


class BenchChatCompletionTaskParams(ChatCompletionTaskParams):
    pass


class PlaceInstanceParams(BaseModel):
    model_id: str
    sharding: Sharding = Sharding.Pipeline
    instance_meta: InstanceMeta = InstanceMeta.MlxRing
    min_nodes: int = 1

    @field_validator("sharding", "instance_meta", mode="plain")
    @classmethod
    def use_default(cls, v: object):
        if not v or not isinstance(v, (Sharding, InstanceMeta)):
            raise PydanticUseDefault()
        return v


class CreateInstanceParams(BaseModel):
    instance: Instance


class PlacementPreview(BaseModel):
    model_id: ModelId
    sharding: Sharding
    instance_meta: InstanceMeta
    instance: Instance | None = None
    # Keys are NodeId strings, values are additional bytes that would be used on that node
    memory_delta_by_node: dict[str, int] | None = None
    error: str | None = None


class PlacementPreviewResponse(BaseModel):
    previews: list[PlacementPreview]


class DeleteInstanceTaskParams(BaseModel):
    instance_id: str


class CreateInstanceResponse(BaseModel):
    message: str
    command_id: CommandId
    model_meta: ModelMetadata


class DeleteInstanceResponse(BaseModel):
    message: str
    command_id: CommandId
    instance_id: InstanceId


class AddCustomModelRequest(BaseModel):
    """Request body for adding a custom model."""

    model_id: str  # HuggingFace repo ID (e.g., "mlx-community/Some-Model-4bit")


class AddCustomModelResponse(BaseModel):
    """Response after adding a custom model."""

    model_id: str  # Full HuggingFace model_id
    name: str
    storage_size_bytes: int
    n_layers: int
    hidden_size: int
    supports_tensor: bool
    is_user_added: bool


class UpdateCustomModelRequest(BaseModel):
    """Request body for updating a custom model."""

    name: str | None = None
    description: str | None = None
    supports_tensor: bool | None = None


class DeleteCustomModelResponse(BaseModel):
    """Response after deleting a custom model."""

    message: str
    model_id: str  # Full HuggingFace model_id that was deleted


class HuggingFaceModelInfo(BaseModel):
    """Search result from HuggingFace Hub."""

    id: str  # Full model ID (e.g., "mlx-community/Llama-3.2-1B-Instruct-4bit")
    author: str
    downloads: int
    likes: int
    last_modified: str
    tags: list[str]
