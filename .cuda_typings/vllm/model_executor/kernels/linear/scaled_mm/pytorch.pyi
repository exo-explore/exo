import abc
import torch
from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel as FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig as FP8ScaledMMLinearLayerConfig,
)
from vllm.config import (
    CompilationMode as CompilationMode,
    get_current_vllm_config as get_current_vllm_config,
)
from vllm.platforms import current_platform as current_platform

class TorchFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel, metaclass=abc.ABCMeta):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]: ...
    def get_output_padding(self) -> int | None: ...

class PerTensorTorchFP8ScaledMMLinearKernel(TorchFP8ScaledMMLinearKernel):
    @classmethod
    def can_implement(
        cls, c: FP8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]: ...
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

class RowWiseTorchFP8ScaledMMLinearKernel(TorchFP8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]: ...
    @classmethod
    def can_implement(
        cls, c: FP8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]: ...
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

class ChannelWiseTorchFP8ScaledMMLinearKernel(TorchFP8ScaledMMLinearKernel):
    @classmethod
    def can_implement(
        cls, c: FP8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]: ...
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
