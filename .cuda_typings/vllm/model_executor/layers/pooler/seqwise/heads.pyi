import abc
import torch
import torch.nn as nn
from .methods import SequencePoolingMethodOutput as SequencePoolingMethodOutput
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Set as Set
from typing import TypeAlias
from vllm.model_executor.layers.pooler import (
    ActivationFn as ActivationFn,
    ClassifierFn as ClassifierFn,
    ProjectorFn as ProjectorFn,
)
from vllm.tasks import PoolingTask as PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata as PoolingMetadata

SequencePoolerHeadOutput: TypeAlias

class SequencePoolerHead(nn.Module, ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    def get_supported_tasks(self) -> Set[PoolingTask]: ...
    @abstractmethod
    def forward(
        self,
        pooled_data: SequencePoolingMethodOutput,
        pooling_metadata: PoolingMetadata,
    ) -> SequencePoolerHeadOutput: ...

class EmbeddingPoolerHead(SequencePoolerHead):
    projector: Incomplete
    head_dtype: Incomplete
    activation: Incomplete
    def __init__(
        self,
        projector: ProjectorFn | None = None,
        head_dtype: torch.dtype | str | None = None,
        activation: ActivationFn | None = None,
    ) -> None: ...
    def get_supported_tasks(self) -> Set[PoolingTask]: ...
    def forward(
        self,
        pooled_data: SequencePoolingMethodOutput,
        pooling_metadata: PoolingMetadata,
    ) -> SequencePoolerHeadOutput: ...

class ClassifierPoolerHead(SequencePoolerHead):
    classifier: Incomplete
    logit_bias: Incomplete
    head_dtype: Incomplete
    activation: Incomplete
    def __init__(
        self,
        classifier: ClassifierFn | None = None,
        logit_bias: float | None = None,
        head_dtype: torch.dtype | str | None = None,
        activation: ActivationFn | None = None,
    ) -> None: ...
    def get_supported_tasks(self) -> Set[PoolingTask]: ...
    def forward(
        self,
        pooled_data: SequencePoolingMethodOutput,
        pooling_metadata: PoolingMetadata,
    ) -> SequencePoolerHeadOutput: ...
