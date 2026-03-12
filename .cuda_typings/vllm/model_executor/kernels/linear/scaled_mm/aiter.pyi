import torch
from .ScaledMMLinearKernel import (
    Int8ScaledMMLinearLayerConfig as Int8ScaledMMLinearLayerConfig,
)
from .cutlass import CutlassInt8ScaledMMLinearKernel as CutlassInt8ScaledMMLinearKernel
from vllm._aiter_ops import rocm_aiter_ops as rocm_aiter_ops
from vllm.platforms import current_platform as current_platform

class AiterInt8ScaledMMLinearKernel(CutlassInt8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]: ...
    @classmethod
    def can_implement(
        cls, c: Int8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]: ...
    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor: ...
