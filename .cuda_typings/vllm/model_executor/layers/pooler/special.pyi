import torch
from .abstract import Pooler, PoolerOutput
from .common import ClassifierFn
from .seqwise import SequencePoolingFn, SequencePoolingMethod
from _typeshed import Incomplete
from collections.abc import Mapping, Set
from vllm.config import PoolerConfig
from vllm.model_executor.layers.pooler import PoolingParamsUpdate
from vllm.tasks import PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata

__all__ = ["BOSEOSFilter", "DispatchPooler", "IdentityPooler"]

class DispatchPooler(Pooler):
    @classmethod
    def for_embedding(cls, pooler_config: PoolerConfig): ...
    @classmethod
    def for_seq_cls(
        cls,
        pooler_config: PoolerConfig,
        *,
        pooling: SequencePoolingMethod | SequencePoolingFn | None = None,
        classifier: ClassifierFn | None = None,
    ): ...
    poolers_by_task: Incomplete
    def __init__(self, poolers_by_task: Mapping[PoolingTask, Pooler]) -> None: ...
    def get_supported_tasks(self) -> Set[PoolingTask]: ...
    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate: ...
    def forward(
        self, hidden_states: torch.Tensor, pooling_metadata: PoolingMetadata
    ) -> PoolerOutput: ...
    def extra_repr(self) -> str: ...

class IdentityPooler(Pooler):
    def get_supported_tasks(self) -> Set[PoolingTask]: ...
    def forward(
        self, hidden_states: torch.Tensor, pooling_metadata: PoolingMetadata
    ) -> PoolerOutput: ...

class BOSEOSFilter(Pooler):
    pooler: Incomplete
    bos_token_id: Incomplete
    eos_token_id: Incomplete
    def __init__(
        self, pooler: Pooler, bos_token_id: int = -1, eos_token_id: int = -1
    ) -> None: ...
    def get_supported_tasks(self) -> Set[PoolingTask]: ...
    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate: ...
    def forward(
        self,
        hidden_states: torch.Tensor | list[torch.Tensor],
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput: ...
