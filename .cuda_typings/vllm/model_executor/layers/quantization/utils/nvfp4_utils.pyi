import torch
from _typeshed import Incomplete
from enum import Enum
from vllm._custom_ops import (
    cutlass_scaled_fp4_mm as cutlass_scaled_fp4_mm,
    cutlass_scaled_mm_supports_fp4 as cutlass_scaled_mm_supports_fp4,
    scaled_fp4_quant as scaled_fp4_quant,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear as apply_fp4_marlin_linear,
    is_fp4_marlin_supported as is_fp4_marlin_supported,
    prepare_fp4_layer_for_marlin as prepare_fp4_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    run_nvfp4_emulations as run_nvfp4_emulations,
)
from vllm.platforms import current_platform as current_platform
from vllm.utils.flashinfer import (
    flashinfer_scaled_fp4_mm as flashinfer_scaled_fp4_mm,
    has_flashinfer as has_flashinfer,
)
from vllm.utils.math_utils import round_up as round_up

logger: Incomplete

class NvFp4LinearBackend(Enum):
    VLLM_CUTLASS = "cutlass"
    FLASHINFER_CUTLASS = "flashinfer-cutlass"
    FLASHINFER_TRTLLM = "flashinfer-trtllm"
    FLASHINFER_CUDNN = "flashinfer-cudnn"
    FBGEMM = "fbgemm"
    MARLIN = "marlin"
    EMULATION = "emulation"

def select_nvfp4_linear_backend() -> NvFp4LinearBackend: ...
def prepare_weights_for_nvfp4_flashinfer_trtllm(
    weight: torch.Tensor, weight_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]: ...
def prepare_weights_for_nvfp4_cutlass(
    weight: torch.Tensor, weight_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int]: ...
def prepare_weights_for_nvfp4_fbgemm(
    weight: torch.Tensor, weight_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]: ...
def convert_to_nvfp4_linear_kernel_format(
    backend: NvFp4LinearBackend, layer: torch.nn.Module
) -> None: ...
def apply_nvfp4_linear(
    backend: NvFp4LinearBackend,
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor: ...
def swizzle_blockscale(scale: torch.Tensor) -> torch.Tensor: ...
def cutlass_fp4_supported() -> bool: ...
def pad_nvfp4_weight_for_cutlass(
    weight: torch.Tensor, alignment: int = 32
) -> tuple[torch.Tensor, int]: ...
def pad_nvfp4_activation_for_cutlass(
    x_fp4: torch.Tensor, weights_padding_bytes: int
) -> torch.Tensor: ...
def slice_nvfp4_output(out: torch.Tensor, output_size: int) -> torch.Tensor: ...
