from pydantic import BaseModel
from typing import TypeAlias
from vllm import PoolingParams as PoolingParams
from vllm.config import ModelConfig as ModelConfig
from vllm.entrypoints.openai.engine.protocol import (
    OpenAIBaseModel as OpenAIBaseModel,
    UsageInfo as UsageInfo,
)
from vllm.entrypoints.pooling.base.protocol import (
    ClassifyRequestMixin as ClassifyRequestMixin,
    PoolingBasicRequestMixin as PoolingBasicRequestMixin,
)
from vllm.entrypoints.pooling.score.utils import (
    ScoreContentPartParam as ScoreContentPartParam,
    ScoreInput as ScoreInput,
    ScoreInputs as ScoreInputs,
)
from vllm.renderers import TokenizeParams as TokenizeParams
from vllm.tasks import PoolingTask as PoolingTask
from vllm.utils import random_uuid as random_uuid

class ScoreRequestMixin(PoolingBasicRequestMixin, ClassifyRequestMixin):
    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams: ...
    def to_pooling_params(self, task: PoolingTask = "score"): ...

class ScoreDataRequest(ScoreRequestMixin):
    data_1: ScoreInputs
    data_2: ScoreInputs

class ScoreQueriesDocumentsRequest(ScoreRequestMixin):
    queries: ScoreInputs
    documents: ScoreInputs
    @property
    def data_1(self): ...
    @property
    def data_2(self): ...

class ScoreQueriesItemsRequest(ScoreRequestMixin):
    queries: ScoreInputs
    items: ScoreInputs
    @property
    def data_1(self): ...
    @property
    def data_2(self): ...

class ScoreTextRequest(ScoreRequestMixin):
    text_1: ScoreInputs
    text_2: ScoreInputs
    @property
    def data_1(self): ...
    @property
    def data_2(self): ...

ScoreRequest: TypeAlias = (
    ScoreQueriesDocumentsRequest
    | ScoreQueriesItemsRequest
    | ScoreDataRequest
    | ScoreTextRequest
)

class RerankRequest(PoolingBasicRequestMixin, ClassifyRequestMixin):
    query: ScoreInput
    documents: ScoreInputs
    top_n: int
    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams: ...
    def to_pooling_params(self, task: PoolingTask = "score"): ...

class RerankDocument(BaseModel):
    text: str | None
    multi_modal: list[ScoreContentPartParam] | None

class RerankResult(BaseModel):
    index: int
    document: RerankDocument
    relevance_score: float

class RerankUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int

class RerankResponse(OpenAIBaseModel):
    id: str
    model: str
    usage: RerankUsage
    results: list[RerankResult]

class ScoreResponseData(OpenAIBaseModel):
    index: int
    object: str
    score: float

class ScoreResponse(OpenAIBaseModel):
    id: str
    object: str
    created: int
    model: str
    data: list[ScoreResponseData]
    usage: UsageInfo
