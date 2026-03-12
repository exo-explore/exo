import torch
from _typeshed import Incomplete
from enum import Enum
from flashinfer.fused_moe.core import ActivationType
from vllm import envs as envs
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation as MoEActivation,
)
from vllm.platforms import current_platform as current_platform
from vllm.utils.math_utils import round_up as round_up

logger: Incomplete

class FlashinferMoeBackend(Enum):
    TENSORRT_LLM = "TensorRT-LLM"
    CUTLASS = "CUTLASS"
    CUTEDSL = "CUTEDSL"

def activation_to_flashinfer_int(activation: MoEActivation) -> int: ...
def activation_to_flashinfer_type(activation: MoEActivation) -> ActivationType: ...
def swap_w13_to_w31(x: torch.Tensor) -> torch.Tensor: ...
def rotate_weights_for_fi_trtllm_fp8_per_tensor_moe(
    gemm1_weights: torch.Tensor, gemm2_weights: torch.Tensor, is_gated_activation: bool
): ...
def get_flashinfer_moe_backend() -> FlashinferMoeBackend: ...
def is_flashinfer_supporting_global_sf(
    backend: FlashinferMoeBackend | None,
) -> bool: ...
def convert_moe_weights_to_flashinfer_trtllm_block_layout(
    cache_permute_indices: dict[torch.Size, torch.Tensor],
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def align_fp4_moe_weights_for_fi(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    is_act_and_mul: bool,
    min_alignment: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]: ...
def align_fp8_moe_weights_for_fi(
    w13: torch.Tensor, w2: torch.Tensor, is_act_and_mul: bool, min_alignment: int = 16
) -> tuple[torch.Tensor, torch.Tensor, int]: ...
def prepare_fp8_moe_layer_for_fi(
    layer: torch.nn.Module,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_input_scale: torch.Tensor | None,
    w2_scale: torch.Tensor,
    w2_input_scale: torch.Tensor | None,
    is_trtllm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
