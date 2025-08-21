import time
from typing import Any, List, Literal

from pydantic import BaseModel, Field

from exo.shared.openai_compat import FinishReason
from exo.shared.types.common import CommandId
from exo.shared.types.models import ModelMetadata
from exo.shared.types.worker.instances import InstanceId


class ModelListModel(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "exo"
    # openwebui fields
    hugging_face_id: str = Field(default="")
    name: str = Field(default="")
    description: str = Field(default="")
    context_length: int = Field(default=0)
    tags: List[str] = Field(default=[])

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelListModel]

class ChatCompletionMessage(BaseModel):
    role: Literal["system", "user", "assistant", "developer", "tool", "function"]
    content: str | None = None
    thinking: str | None = None  # Added for GPT-OSS harmony format support
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    function_call: dict[str, Any] | None = None


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


class StreamingChoiceResponse(BaseModel):
    index: int
    delta: ChatCompletionMessage
    logprobs: Logprobs | None = None
    finish_reason: FinishReason | None = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    logprobs: Logprobs | None = None
    finish_reason: FinishReason | None = None


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



class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice | StreamingChoiceResponse]
    usage: Usage | None = None
    service_tier: str | None = None


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

class CreateInstanceTaskParams(BaseModel):
    # TODO: in future the user could specify a specific Instance, not just a model_id
    model_id: str

class DeleteInstanceTaskParams(BaseModel):
    instance_id: str

class CreateInstanceResponse(BaseModel):
    message: str
    command_id: CommandId
    model_meta: ModelMetadata
    instance_id: InstanceId

class DeleteInstanceResponse(BaseModel):
    message: str
    command_id: CommandId
    instance_id: InstanceId
