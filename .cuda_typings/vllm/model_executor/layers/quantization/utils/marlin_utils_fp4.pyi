import torch
from _typeshed import Incomplete
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    USE_FP32_REDUCE_DEFAULT as USE_FP32_REDUCE_DEFAULT,
    get_marlin_input_dtype as get_marlin_input_dtype,
    marlin_make_workspace_new as marlin_make_workspace_new,
    marlin_permute_bias as marlin_permute_bias,
    marlin_permute_scales as marlin_permute_scales,
    marlin_quant_input as marlin_quant_input,
    should_use_atomic_add_reduce as should_use_atomic_add_reduce,
)
from vllm.platforms import current_platform as current_platform
from vllm.scalar_type import scalar_types as scalar_types

FP4_MARLIN_SUPPORTED_GROUP_SIZES: Incomplete
logger: Incomplete

def is_fp4_marlin_supported(): ...
def nvfp4_marlin_process_scales(marlin_scales): ...
def mxfp4_marlin_process_scales(marlin_scales, input_dtype=None): ...
def nvfp4_marlin_process_global_scale(global_scale): ...
def apply_fp4_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor | None,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
    bias: torch.Tensor | None = None,
    input_dtype: torch.dtype | None = None,
    use_fp32_reduce: bool = ...,
) -> torch.Tensor: ...
def prepare_fp4_layer_for_marlin(
    layer: torch.nn.Module, input_dtype: torch.dtype | None = None
) -> None: ...
def prepare_nvfp4_moe_layer_for_marlin(
    layer: torch.nn.Module,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_scale_2: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_scale_2: torch.Tensor,
    is_act_and_mul: bool,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]: ...
def prepare_moe_fp4_layer_for_marlin(
    layer: torch.nn.Module, input_dtype: torch.dtype | None = None
) -> None: ...
def rand_marlin_weight_nvfp4_like(weight, group_size, input_dtype=None): ...
def rand_marlin_weight_mxfp4_like(weight, group_size, input_dtype=None): ...
