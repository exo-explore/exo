from typing import TypeAlias
from vllm import PoolingParams as PoolingParams
from vllm.config import ModelConfig as ModelConfig
from vllm.entrypoints.openai.engine.protocol import (
    OpenAIBaseModel as OpenAIBaseModel,
    UsageInfo as UsageInfo,
)
from vllm.entrypoints.pooling.base.protocol import (
    ChatRequestMixin as ChatRequestMixin,
    CompletionRequestMixin as CompletionRequestMixin,
    EmbedRequestMixin as EmbedRequestMixin,
    PoolingBasicRequestMixin as PoolingBasicRequestMixin,
)
from vllm.renderers import TokenizeParams as TokenizeParams
from vllm.utils import random_uuid as random_uuid

class EmbeddingCompletionRequest(
    PoolingBasicRequestMixin, CompletionRequestMixin, EmbedRequestMixin
):
    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams: ...
    def to_pooling_params(self): ...

class EmbeddingChatRequest(
    PoolingBasicRequestMixin, ChatRequestMixin, EmbedRequestMixin
):
    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams: ...
    def to_pooling_params(self): ...

EmbeddingRequest: TypeAlias = EmbeddingCompletionRequest | EmbeddingChatRequest

class EmbeddingResponseData(OpenAIBaseModel):
    index: int
    object: str
    embedding: list[float] | str

class EmbeddingResponse(OpenAIBaseModel):
    id: str
    object: str
    created: int
    model: str
    data: list[EmbeddingResponseData]
    usage: UsageInfo

class EmbeddingBytesResponse(OpenAIBaseModel):
    content: list[bytes]
    headers: dict[str, str] | None
    media_type: str
