import torch
from .heads import (
    ClassifierPoolerHead as ClassifierPoolerHead,
    EmbeddingPoolerHead as EmbeddingPoolerHead,
    SequencePoolerHead as SequencePoolerHead,
    SequencePoolerHeadOutput as SequencePoolerHeadOutput,
)
from .methods import (
    SequencePoolingMethod as SequencePoolingMethod,
    SequencePoolingMethodOutput as SequencePoolingMethodOutput,
    get_seq_pooling_method as get_seq_pooling_method,
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
)
from vllm.model_executor.layers.pooler.abstract import Pooler as Pooler
from vllm.model_executor.layers.pooler.activations import (
    PoolerActivation as PoolerActivation,
    PoolerNormalize as PoolerNormalize,
    resolve_classifier_act_fn as resolve_classifier_act_fn,
)
from vllm.tasks import POOLING_TASKS as POOLING_TASKS, PoolingTask as PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata as PoolingMetadata

SequencePoolingFn: TypeAlias
SequencePoolingHeadFn: TypeAlias = Callable[
    [SequencePoolingMethodOutput, PoolingMetadata], SequencePoolerHeadOutput
]
SequencePoolerOutput: TypeAlias

class SequencePooler(Pooler):
    pooling: Incomplete
    head: Incomplete
    def __init__(
        self,
        pooling: SequencePoolingMethod | SequencePoolingFn,
        head: SequencePoolerHead | SequencePoolingHeadFn,
    ) -> None: ...
    def get_supported_tasks(self) -> Set[PoolingTask]: ...
    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate: ...
    def forward(
        self, hidden_states: torch.Tensor, pooling_metadata: PoolingMetadata
    ) -> SequencePoolerOutput: ...

def pooler_for_embed(pooler_config: PoolerConfig): ...
def pooler_for_classify(
    pooler_config: PoolerConfig,
    *,
    pooling: SequencePoolingMethod | SequencePoolingFn | None = None,
    classifier: ClassifierFn | None = None,
    act_fn: PoolerActivation | str | None = None,
): ...
