import abc
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from collections.abc import Set as Set
from typing import TypeAlias
from vllm.config.pooler import SequencePoolingType as SequencePoolingType
from vllm.model_executor.layers.pooler import PoolingParamsUpdate as PoolingParamsUpdate
from vllm.tasks import PoolingTask as PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata as PoolingMetadata

SequencePoolingMethodOutput: TypeAlias

class SequencePoolingMethod(nn.Module, ABC, metaclass=abc.ABCMeta):
    def get_supported_tasks(self) -> Set[PoolingTask]: ...
    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate: ...
    @abstractmethod
    def forward(
        self, hidden_states: torch.Tensor, pooling_metadata: PoolingMetadata
    ) -> SequencePoolingMethodOutput: ...

class CLSPool(SequencePoolingMethod):
    def forward(
        self, hidden_states: torch.Tensor, pooling_metadata: PoolingMetadata
    ) -> SequencePoolingMethodOutput: ...

class LastPool(SequencePoolingMethod):
    def forward(
        self, hidden_states: torch.Tensor, pooling_metadata: PoolingMetadata
    ) -> SequencePoolingMethodOutput: ...

class MeanPool(SequencePoolingMethod):
    def forward(
        self, hidden_states: torch.Tensor, pooling_metadata: PoolingMetadata
    ) -> SequencePoolingMethodOutput: ...

def get_seq_pooling_method(pooling_type: SequencePoolingType | str): ...
