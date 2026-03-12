import torch
from _typeshed import Incomplete
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    USE_FP32_REDUCE_DEFAULT as USE_FP32_REDUCE_DEFAULT,
    get_marlin_input_dtype as get_marlin_input_dtype,
    marlin_make_workspace_new as marlin_make_workspace_new,
    marlin_permute_bias as marlin_permute_bias,
    marlin_permute_scales as marlin_permute_scales,
    should_use_atomic_add_reduce as should_use_atomic_add_reduce,
)
from vllm.model_executor.utils import replace_parameter as replace_parameter
from vllm.platforms import current_platform as current_platform
from vllm.scalar_type import scalar_types as scalar_types

logger: Incomplete

def is_fp8_marlin_supported(): ...
def fp8_fused_exponent_bias_into_scales(scales): ...
def apply_fp8_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
    bias: torch.Tensor | None,
    input_dtype: torch.dtype | None = None,
    use_fp32_reduce: bool = ...,
) -> torch.Tensor: ...
def prepare_fp8_layer_for_marlin(
    layer: torch.nn.Module,
    size_k_first: bool = True,
    input_dtype: torch.dtype | None = None,
) -> None: ...
def prepare_fp8_moe_layer_for_marlin(
    layer: torch.nn.Module,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...
def pack_fp8_to_int32(
    fp8_tensor: torch.Tensor, size_k_first: bool = True
) -> torch.Tensor: ...
def marlin_quant_fp8_torch(weight, group_size, input_dtype=None): ...
