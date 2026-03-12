import abc
import torch
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Callable as Callable
from dataclasses import dataclass
from vllm.model_executor.layers.quantization.utils import (
    replace_parameter as replace_parameter,
)
from vllm.scalar_type import ScalarType as ScalarType

@dataclass
class MPLinearLayerConfig:
    full_weight_shape: tuple[int, int]
    partition_weight_shape: tuple[int, int]
    weight_type: ScalarType
    act_type: torch.dtype
    group_size: int
    zero_points: bool
    has_g_idx: bool
    out_type: torch.dtype | None = ...

class MPLinearKernel(ABC, metaclass=abc.ABCMeta):
    @classmethod
    @abstractmethod
    def get_min_capability(cls) -> int: ...
    @classmethod
    @abstractmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]: ...
    config: Incomplete
    w_q_name: Incomplete
    w_s_name: Incomplete
    w_zp_name: Incomplete
    w_gidx_name: Incomplete
    def __init__(
        self,
        c: MPLinearLayerConfig,
        w_q_param_name: str,
        w_s_param_name: str,
        w_zp_param_name: str | None = None,
        w_gidx_param_name: str | None = None,
    ) -> None: ...
    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None: ...
    @abstractmethod
    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor: ...
