import torch
from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel as FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig as FP8ScaledMMLinearLayerConfig,
)
from vllm.platforms import current_platform as current_platform
from vllm.utils.flashinfer import (
    flashinfer_scaled_fp8_mm as flashinfer_scaled_fp8_mm,
    has_flashinfer as has_flashinfer,
)

class FlashInferFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
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
