import torch
from .heads import (
    TokenClassifierPoolerHead as TokenClassifierPoolerHead,
    TokenEmbeddingPoolerHead as TokenEmbeddingPoolerHead,
    TokenPoolerHead as TokenPoolerHead,
    TokenPoolerHeadOutputItem as TokenPoolerHeadOutputItem,
)
from .methods import (
    TokenPoolingMethod as TokenPoolingMethod,
    TokenPoolingMethodOutputItem as TokenPoolingMethodOutputItem,
    get_tok_pooling_method as get_tok_pooling_method,
)
from _typeshed import Incomplete
from collections.abc import Callable, Set as Set
from typing import TypeAlias
from vllm.config import (
    PoolerConfig as PoolerConfig,
    get_current_vllm_config as get_current_vllm_config,
)
from vllm.model_executor.layers.pooler import (
    ClassifierFn as ClassifierFn,
    PoolingParamsUpdate as PoolingParamsUpdate,
    ProjectorFn as ProjectorFn,
)
from vllm.model_executor.layers.pooler.abstract import Pooler as Pooler
from vllm.model_executor.layers.pooler.activations import (
    PoolerActivation as PoolerActivation,
    PoolerNormalize as PoolerNormalize,
    resolve_classifier_act_fn as resolve_classifier_act_fn,
)
from vllm.tasks import POOLING_TASKS as POOLING_TASKS, PoolingTask as PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata as PoolingMetadata

TokenPoolingFn: TypeAlias
TokenPoolingHeadFn: TypeAlias = Callable[
    [list[TokenPoolingMethodOutputItem], PoolingMetadata],
    list[TokenPoolerHeadOutputItem],
]
TokenPoolerOutput: TypeAlias

class TokenPooler(Pooler):
    pooling: Incomplete
    head: Incomplete
    def __init__(
        self,
        pooling: TokenPoolingMethod | TokenPoolingFn,
        head: TokenPoolerHead | TokenPoolingHeadFn,
    ) -> None: ...
    def get_supported_tasks(self) -> Set[PoolingTask]: ...
    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate: ...
    def forward(
        self, hidden_states: torch.Tensor, pooling_metadata: PoolingMetadata
    ) -> TokenPoolerOutput: ...

def pooler_for_token_embed(
    pooler_config: PoolerConfig, projector: ProjectorFn | None = None
) -> TokenPooler: ...
def pooler_for_token_classify(
    pooler_config: PoolerConfig,
    *,
    pooling: TokenPoolingMethod | TokenPoolingFn | None = None,
    classifier: ClassifierFn | None = None,
    act_fn: PoolerActivation | str | None = None,
): ...
