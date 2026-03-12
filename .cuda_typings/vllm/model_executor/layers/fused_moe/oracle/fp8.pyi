import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from _typeshed import Incomplete
from enum import Enum
from vllm import envs as envs
from vllm._aiter_ops import rocm_aiter_ops as rocm_aiter_ops
from vllm.config.kernel import MoEBackend as MoEBackend
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize as maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig as FusedMoEConfig,
    FusedMoEQuantConfig as FusedMoEQuantConfig,
    fp8_w8a16_moe_quant_config as fp8_w8a16_moe_quant_config,
    fp8_w8a8_moe_quant_config as fp8_w8a8_moe_quant_config,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    FlashinferMoeBackend as FlashinferMoeBackend,
    get_flashinfer_moe_backend as get_flashinfer_moe_backend,
    prepare_fp8_moe_layer_for_fi as prepare_fp8_moe_layer_for_fi,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    prepare_fp8_moe_layer_for_deepgemm as prepare_fp8_moe_layer_for_deepgemm,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    prepare_fp8_moe_layer_for_marlin as prepare_fp8_moe_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey as QuantKey,
    kFp8Dynamic128Sym as kFp8Dynamic128Sym,
    kFp8Static128BlockSym as kFp8Static128BlockSym,
)
from vllm.platforms import current_platform as current_platform

logger: Incomplete

class Fp8MoeBackend(Enum):
    NONE = "NONE"
    FLASHINFER_TRTLLM = "FLASHINFER_TRTLLM"
    FLASHINFER_CUTLASS = "FLASHINFER_CUTLASS"
    DEEPGEMM = "DEEPGEMM"
    BATCHED_DEEPGEMM = "BATCHED_DEEPGEMM"
    MARLIN = "MARLIN"
    TRITON = "TRITON"
    BATCHED_TRITON = "BATCHED_TRITON"
    AITER = "AITER"
    VLLM_CUTLASS = "VLLM_CUTLASS"
    BATCHED_VLLM_CUTLASS = "BATCHED_VLLM_CUTLASS"
    XPU = "XPU"

def backend_to_kernel_cls(backend: Fp8MoeBackend) -> type[mk.FusedMoEExperts]: ...
def map_fp8_backend(runner_backend: MoEBackend) -> Fp8MoeBackend: ...
def select_fp8_moe_backend(
    config: FusedMoEConfig,
    weight_key: QuantKey | None,
    activation_key: QuantKey | None,
    allow_vllm_cutlass: bool = False,
) -> tuple[Fp8MoeBackend, type[mk.FusedMoEExperts] | None]: ...
def convert_to_fp8_moe_kernel_format(
    fp8_backend: Fp8MoeBackend,
    layer: torch.nn.Module,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_input_scale: torch.Tensor | None,
    w2_input_scale: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...
def make_fp8_moe_quant_config(
    fp8_backend: Fp8MoeBackend,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    a1_scale: torch.Tensor | None,
    a2_scale: torch.Tensor | None,
    block_shape: list[int] | None = None,
    per_act_token_quant: bool = False,
    per_out_ch_quant: bool = False,
) -> FusedMoEQuantConfig: ...
def make_fp8_moe_kernel(
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    experts_cls: type[mk.FusedMoEExperts],
    fp8_backend: Fp8MoeBackend,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    shared_experts: torch.nn.Module | None = None,
) -> mk.FusedMoEKernel: ...
