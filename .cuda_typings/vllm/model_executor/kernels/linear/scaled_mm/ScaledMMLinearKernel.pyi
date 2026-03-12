import abc
import torch
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8 as QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey as QuantKey,
)
from vllm.platforms import current_platform as current_platform

@dataclass
class ScaledMMLinearLayerConfig: ...

@dataclass
class Int8ScaledMMLinearLayerConfig(ScaledMMLinearLayerConfig):
    is_static_input_scheme: bool
    is_channelwise: bool
    input_symmetric: bool

@dataclass
class FP8ScaledMMLinearLayerConfig(ScaledMMLinearLayerConfig):
    weight_quant_key: QuantKey
    activation_quant_key: QuantKey
    out_dtype: torch.dtype | None

class ScaledMMLinearKernel(ABC, Generic[_ConfigT, _ParamsT], metaclass=abc.ABCMeta):
    @classmethod
    @abstractmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]: ...
    @classmethod
    @abstractmethod
    def can_implement(cls, c: _ConfigT) -> tuple[bool, str | None]: ...
    config: Incomplete
    layer_param_names: Incomplete
    def __init__(self, c: _ConfigT, layer_param_names: Sequence[str]) -> None: ...
    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None: ...
    @abstractmethod
    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor: ...

class FP8ScaledMMLinearKernel(
    ScaledMMLinearKernel[FP8ScaledMMLinearLayerConfig, _FP8ParamsT],
    ABC,
    metaclass=abc.ABCMeta,
):
    quant_fp8: Incomplete
    fp8_dtype: Incomplete
    def __init__(
        self, c: FP8ScaledMMLinearLayerConfig, layer_param_names: Sequence[str]
    ) -> None: ...
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None: ...
    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor: ...
    @abstractmethod
    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor: ...
    def get_output_padding(self) -> int | None: ...

class Int8ScaledMMLinearKernel(
    ScaledMMLinearKernel[Int8ScaledMMLinearLayerConfig, _Int8ParamsT],
    ABC,
    metaclass=abc.ABCMeta,
): ...
