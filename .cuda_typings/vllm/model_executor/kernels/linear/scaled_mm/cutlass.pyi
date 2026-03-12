import torch
from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel as FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig as FP8ScaledMMLinearLayerConfig,
    Int8ScaledMMLinearKernel as Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig as Int8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils import (
    replace_parameter as replace_parameter,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise as convert_to_channelwise,
)
from vllm.platforms import current_platform as current_platform

class CutlassInt8ScaledMMLinearKernel(Int8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]: ...
    @classmethod
    def can_implement(
        cls, c: Int8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]: ...
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None: ...
    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor: ...

class CutlassFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
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
