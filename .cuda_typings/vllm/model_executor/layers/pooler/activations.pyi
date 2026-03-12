import abc
import torch
import torch.nn as nn
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Callable as Callable
from transformers import PretrainedConfig as PretrainedConfig
from vllm.config import (
    ModelConfig as ModelConfig,
    get_current_vllm_config as get_current_vllm_config,
)
from vllm.logger import init_logger as init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname as resolve_obj_by_qualname

logger: Incomplete

def get_classification_act_fn(config: PretrainedConfig) -> PoolerActivation: ...
def get_cross_encoder_act_fn(config: PretrainedConfig) -> PoolerActivation: ...
def resolve_classifier_act_fn(
    model_config: ModelConfig,
    static_num_labels: bool = True,
    act_fn: PoolerActivation | str | None = None,
): ...

class PoolerActivation(nn.Module, ABC, metaclass=abc.ABCMeta):
    @staticmethod
    def wraps(module: nn.Module): ...
    @abstractmethod
    def forward_chunk(self, pooled_data: torch.Tensor) -> torch.Tensor: ...
    def forward(self, pooled_data: _T) -> _T: ...

class PoolerIdentity(PoolerActivation):
    def forward_chunk(self, pooled_data: torch.Tensor) -> torch.Tensor: ...

class PoolerNormalize(PoolerActivation):
    def forward_chunk(self, pooled_data: torch.Tensor) -> torch.Tensor: ...

class PoolerMultiLabelClassify(PoolerActivation):
    def forward_chunk(self, pooled_data: torch.Tensor) -> torch.Tensor: ...

class PoolerClassify(PoolerActivation):
    num_labels: Incomplete
    def __init__(self, *, static_num_labels: bool = True) -> None: ...
    def forward_chunk(self, pooled_data: torch.Tensor) -> torch.Tensor: ...

class LambdaPoolerActivation(PoolerActivation):
    fn: Incomplete
    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> None: ...
    def forward_chunk(self, pooled_data: torch.Tensor) -> torch.Tensor: ...
