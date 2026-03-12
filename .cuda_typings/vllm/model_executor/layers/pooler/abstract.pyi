import abc
import torch
import torch.nn as nn
from .common import PoolingParamsUpdate
from abc import ABC, abstractmethod
from collections.abc import Set
from vllm.tasks import PoolingTask
from vllm.v1.outputs import PoolerOutput
from vllm.v1.pool.metadata import PoolingMetadata

__all__ = ["Pooler"]

class Pooler(nn.Module, ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    def get_supported_tasks(self) -> Set[PoolingTask]: ...
    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate: ...
    @abstractmethod
    def forward(
        self, hidden_states: torch.Tensor, pooling_metadata: PoolingMetadata
    ) -> PoolerOutput: ...
