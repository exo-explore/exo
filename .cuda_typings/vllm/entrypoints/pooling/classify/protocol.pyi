from _typeshed import Incomplete
from typing import TypeAlias
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
    PoolingBasicRequestMixin as PoolingBasicRequestMixin,
)
from vllm.logger import init_logger as init_logger
from vllm.renderers import TokenizeParams as TokenizeParams
from vllm.utils import random_uuid as random_uuid

logger: Incomplete

class ClassificationCompletionRequest(
    PoolingBasicRequestMixin, CompletionRequestMixin, ClassifyRequestMixin
):
    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams: ...
    def to_pooling_params(self): ...

class ClassificationChatRequest(
    PoolingBasicRequestMixin, ChatRequestMixin, ClassifyRequestMixin
):
    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams: ...
    def to_pooling_params(self): ...

ClassificationRequest: TypeAlias = (
    ClassificationCompletionRequest | ClassificationChatRequest
)

class ClassificationData(OpenAIBaseModel):
    index: int
    label: str | None
    probs: list[float]
    num_classes: int

class ClassificationResponse(OpenAIBaseModel):
    id: str
    object: str
    created: int
    model: str
    data: list[ClassificationData]
    usage: UsageInfo
