import torch
from .ScaledMMLinearKernel import (
    Int8ScaledMMLinearLayerConfig as Int8ScaledMMLinearLayerConfig,
)
from .cutlass import CutlassInt8ScaledMMLinearKernel as CutlassInt8ScaledMMLinearKernel
from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (
    triton_scaled_mm as triton_scaled_mm,
)
from vllm.model_executor.layers.quantization.utils import (
    replace_parameter as replace_parameter,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise as convert_to_channelwise,
)
from vllm.platforms import current_platform as current_platform

class TritonInt8ScaledMMLinearKernel(CutlassInt8ScaledMMLinearKernel):
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
