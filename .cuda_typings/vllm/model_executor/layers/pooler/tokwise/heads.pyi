import abc
import torch
import torch.nn as nn
from .methods import TokenPoolingMethodOutputItem as TokenPoolingMethodOutputItem
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Set as Set
from typing import TypeAlias
from vllm.model_executor.layers.pooler import (
    ActivationFn as ActivationFn,
    ClassifierFn as ClassifierFn,
    ProjectorFn as ProjectorFn,
)
from vllm.pooling_params import PoolingParams as PoolingParams
from vllm.tasks import PoolingTask as PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata as PoolingMetadata

TokenPoolerHeadOutputItem: TypeAlias

class TokenPoolerHead(nn.Module, ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    def get_supported_tasks(self) -> Set[PoolingTask]: ...
    @abstractmethod
    def forward_chunk(
        self, pooled_data: TokenPoolingMethodOutputItem, pooling_param: PoolingParams
    ) -> TokenPoolerHeadOutputItem: ...
    def forward(
        self,
        pooled_data: list[TokenPoolingMethodOutputItem],
        pooling_metadata: PoolingMetadata,
    ) -> list[TokenPoolerHeadOutputItem]: ...

class TokenEmbeddingPoolerHead(TokenPoolerHead):
    head_dtype: Incomplete
    projector: Incomplete
    activation: Incomplete
    def __init__(
        self,
        head_dtype: torch.dtype | str | None = None,
        projector: ProjectorFn | None = None,
        activation: ActivationFn | None = None,
    ) -> None: ...
    def get_supported_tasks(self) -> Set[PoolingTask]: ...
    def forward_chunk(
        self, pooled_data: TokenPoolingMethodOutputItem, pooling_param: PoolingParams
    ) -> TokenPoolerHeadOutputItem: ...

class TokenClassifierPoolerHead(TokenPoolerHead):
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
    def forward_chunk(
        self, pooled_data: TokenPoolingMethodOutputItem, pooling_param: PoolingParams
    ) -> TokenPoolerHeadOutputItem: ...
