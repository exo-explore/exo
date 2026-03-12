from typing import Generic, TypeAlias, TypeVar
from vllm import PoolingParams as PoolingParams
from vllm.config import ModelConfig as ModelConfig
from vllm.entrypoints.openai.engine.protocol import (
    OpenAIBaseModel as OpenAIBaseModel,
    UsageInfo as UsageInfo,
)
from vllm.entrypoints.pooling.base.protocol import (
    ChatRequestMixin as ChatRequestMixin,
    ClassifyRequestMixin as ClassifyRequestMixin,
    CompletionRequestMixin as CompletionRequestMixin,
    EmbedRequestMixin as EmbedRequestMixin,
    EncodingRequestMixin as EncodingRequestMixin,
    PoolingBasicRequestMixin as PoolingBasicRequestMixin,
)
from vllm.renderers import TokenizeParams as TokenizeParams
from vllm.tasks import PoolingTask as PoolingTask
from vllm.utils import random_uuid as random_uuid

class PoolingCompletionRequest(
    PoolingBasicRequestMixin,
    CompletionRequestMixin,
    EmbedRequestMixin,
    ClassifyRequestMixin,
):
    task: PoolingTask | None
    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams: ...
    def to_pooling_params(self): ...

class PoolingChatRequest(
    PoolingBasicRequestMixin, ChatRequestMixin, EmbedRequestMixin, ClassifyRequestMixin
):
    task: PoolingTask | None
    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams: ...
    def to_pooling_params(self): ...

T = TypeVar("T")

class IOProcessorRequest(PoolingBasicRequestMixin, EncodingRequestMixin, Generic[T]):
    data: T
    task: PoolingTask
    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams: ...

class IOProcessorResponse(OpenAIBaseModel, Generic[T]):
    request_id: str | None
    created_at: int
    data: T

PoolingRequest: TypeAlias = (
    PoolingCompletionRequest | PoolingChatRequest | IOProcessorRequest
)

class PoolingResponseData(OpenAIBaseModel):
    index: int
    object: str
    data: list[list[float]] | list[float] | str

class PoolingResponse(OpenAIBaseModel):
    id: str
    object: str
    created: int
    model: str
    data: list[PoolingResponseData]
    usage: UsageInfo

class PoolingBytesResponse(OpenAIBaseModel):
    content: list[bytes]
    headers: dict[str, str] | None
    media_type: str
