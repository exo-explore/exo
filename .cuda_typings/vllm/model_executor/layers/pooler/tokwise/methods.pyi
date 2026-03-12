import abc
import torch
import torch.nn as nn
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Set as Set
from typing import TypeAlias
from vllm.config import get_current_vllm_config as get_current_vllm_config
from vllm.config.pooler import TokenPoolingType as TokenPoolingType
from vllm.model_executor.layers.pooler import PoolingParamsUpdate as PoolingParamsUpdate
from vllm.tasks import PoolingTask as PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata as PoolingMetadata

TokenPoolingMethodOutputItem: TypeAlias

class TokenPoolingMethod(nn.Module, ABC, metaclass=abc.ABCMeta):
    def get_supported_tasks(self) -> Set[PoolingTask]: ...
    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate: ...
    @abstractmethod
    def forward(
        self, hidden_states: torch.Tensor, pooling_metadata: PoolingMetadata
    ) -> list[TokenPoolingMethodOutputItem]: ...

class AllPool(TokenPoolingMethod):
    enable_chunked_prefill: Incomplete
    def __init__(self) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, pooling_metadata: PoolingMetadata
    ) -> list[TokenPoolingMethodOutputItem]: ...

class StepPool(AllPool):
    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate: ...
    def forward(
        self, hidden_states: torch.Tensor, pooling_metadata: PoolingMetadata
    ) -> list[TokenPoolingMethodOutputItem]: ...

def get_tok_pooling_method(pooling_type: TokenPoolingType | str): ...
