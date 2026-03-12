import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from _typeshed import Incomplete
from enum import Enum
from vllm.config.kernel import MoEBackend as MoEBackend
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize as maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig as FusedMoEConfig,
    FusedMoEQuantConfig as FusedMoEQuantConfig,
    mxfp4_w4a16_moe_quant_config as mxfp4_w4a16_moe_quant_config,
    nvfp4_moe_quant_config as nvfp4_moe_quant_config,
    nvfp4_w4a16_moe_quant_config as nvfp4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
    prepare_nvfp4_moe_layer_for_fi_or_cutlass as prepare_nvfp4_moe_layer_for_fi_or_cutlass,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    FlashinferMoeBackend as FlashinferMoeBackend,
    get_flashinfer_moe_backend as get_flashinfer_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    prepare_nvfp4_moe_layer_for_marlin as prepare_nvfp4_moe_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey as QuantKey,
)

logger: Incomplete

class NvFp4MoeBackend(Enum):
    FLASHINFER_TRTLLM = "FLASHINFER_TRTLLM"
    FLASHINFER_CUTLASS = "FLASHINFER_CUTLASS"
    FLASHINFER_CUTEDSL = "FLASHINFER_CUTEDSL"
    VLLM_CUTLASS = "VLLM_CUTLASS"
    MARLIN = "MARLIN"

FLASHINFER_NVFP4_MOE_BACKENDS: Incomplete
fi_2_vllm_backend_map: dict[FlashinferMoeBackend, NvFp4MoeBackend]

def is_global_sf_supported_for_nvfp4_backend(backend: NvFp4MoeBackend) -> bool: ...
def backend_to_kernel_cls(
    backend: NvFp4MoeBackend,
) -> list[type[mk.FusedMoEExperts]]: ...
def map_nvfp4_backend(runner_backend: MoEBackend) -> NvFp4MoeBackend: ...
def select_nvfp4_moe_backend(
    config: FusedMoEConfig, weight_key: QuantKey | None, activation_key: QuantKey | None
) -> tuple[NvFp4MoeBackend, type[mk.FusedMoEExperts]]: ...
def convert_to_nvfp4_moe_kernel_format(
    nvfp4_backend: NvFp4MoeBackend,
    layer: torch.nn.Module,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_scale_2: torch.Tensor,
    a13_scale: torch.Tensor | None,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_scale_2: torch.Tensor,
    a2_scale: torch.Tensor | None,
    is_act_and_mul: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]: ...
def make_mxfp4_moe_quant_config(
    w13_scale: torch.Tensor, w2_scale: torch.Tensor
) -> FusedMoEQuantConfig: ...
def make_nvfp4_moe_quant_config(
    backend: NvFp4MoeBackend,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_scale_2: torch.Tensor,
    w2_scale_2: torch.Tensor,
    a13_scale: torch.Tensor,
    a2_scale: torch.Tensor,
) -> FusedMoEQuantConfig: ...
def make_nvfp4_moe_kernel(
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    experts_cls: type[mk.FusedMoEExperts],
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    shared_experts: torch.nn.Module | None = None,
) -> mk.FusedMoEKernel: ...
