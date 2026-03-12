import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from _typeshed import Incomplete
from enum import Enum
from torch.nn import Module as Module
from vllm._aiter_ops import rocm_aiter_ops as rocm_aiter_ops
from vllm.config.kernel import MoEBackend as MoEBackend
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig as FusedMoEConfig,
    FusedMoEQuantConfig as FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.flashinfer_trtllm_moe import (
    is_supported_config_trtllm_bf16 as is_supported_config_trtllm_bf16,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoDPEPModular as MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    swap_w13_to_w31 as swap_w13_to_w31,
)
from vllm.platforms import current_platform as current_platform
from vllm.utils.flashinfer import (
    has_flashinfer as has_flashinfer,
    has_flashinfer_cutlass_fused_moe as has_flashinfer_cutlass_fused_moe,
)

logger: Incomplete

class UnquantizedMoeBackend(Enum):
    FLASHINFER_TRTLLM = "FlashInfer TRTLLM"
    FLASHINFER_CUTLASS = "FlashInfer CUTLASS"
    AITER = "ROCm AITER"
    TRITON = "TRITON"
    CPU = "CPU"
    XPU = "XPU"
    TPU = "TPU"
    OOT = "OOT"

UNSUPPORTED_BACKEND: Incomplete

def map_unquantized_backend(runner_backend: MoEBackend) -> UnquantizedMoeBackend: ...
def select_unquantized_moe_backend(
    moe_config: FusedMoEConfig, use_ep: bool, use_dp: bool
) -> UnquantizedMoeBackend: ...
def convert_to_unquantized_kernel_format(
    unquantized_backend: UnquantizedMoeBackend,
    layer: Module,
    w13_weight: torch.Tensor | None = None,
    w2_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def make_unquantized_moe_kernel(
    backend: UnquantizedMoeBackend,
    quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
) -> mk.FusedMoEKernel | None: ...
