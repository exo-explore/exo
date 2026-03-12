import torch
from _typeshed import Incomplete
from collections.abc import Sequence
from vllm.model_executor.kernels.linear import (
    FP8ScaledMMLinearKernel as FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig as FP8ScaledMMLinearLayerConfig,
)
from vllm.platforms import current_platform as current_platform

class XPUFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]: ...
    @classmethod
    def can_implement(
        cls, c: FP8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]: ...
    config: Incomplete
    layer_param_names: Incomplete
    def __init__(
        self, c: FP8ScaledMMLinearLayerConfig, layer_param_names: Sequence[str]
    ) -> None: ...
    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor: ...
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
