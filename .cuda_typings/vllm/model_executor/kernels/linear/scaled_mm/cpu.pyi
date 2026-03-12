import torch
from .ScaledMMLinearKernel import (
    Int8ScaledMMLinearKernel as Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig as Int8ScaledMMLinearLayerConfig,
)
from _typeshed import Incomplete
from vllm import envs as envs
from vllm.model_executor.layers.quantization.utils import (
    replace_parameter as replace_parameter,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise as convert_to_channelwise,
)
from vllm.model_executor.layers.utils import (
    check_cpu_sgl_kernel as check_cpu_sgl_kernel,
)
from vllm.platforms import current_platform as current_platform
from vllm.platforms.interface import CpuArchEnum as CpuArchEnum

class CPUInt8ScaledMMLinearKernel(Int8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]: ...
    @classmethod
    def can_implement(
        cls, c: Int8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]: ...
    linear_method: Incomplete
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None: ...
    dnnl_handler: Incomplete
    def process_weights_for_onednn(self, layer: torch.nn.Module) -> None: ...
    def process_weights_for_sgl(self, layer: torch.nn.Module) -> None: ...
    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor: ...
