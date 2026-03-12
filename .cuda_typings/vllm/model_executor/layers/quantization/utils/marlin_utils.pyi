import torch
from .quant_utils import pack_cols as pack_cols, unpack_cols as unpack_cols
from _typeshed import Incomplete
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.linear import LinearBase as LinearBase
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8 as QuantFP8
from vllm.model_executor.layers.quantization.utils.int8_utils import (
    per_token_quant_int8 as per_token_quant_int8,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape as GroupShape,
)
from vllm.platforms import current_platform as current_platform
from vllm.scalar_type import ScalarType as ScalarType, scalar_types as scalar_types
from vllm.utils.platform_utils import num_compute_units as num_compute_units

logger: Incomplete
GPTQ_MARLIN_TILE: int
GPTQ_MARLIN_MIN_THREAD_N: int
GPTQ_MARLIN_MIN_THREAD_K: int
GPTQ_MARLIN_MAX_PARALLEL: int
MARLIN_SUPPORTED_GROUP_SIZES: Incomplete
USE_FP32_REDUCE_DEFAULT: bool

def query_marlin_supported_quant_types(
    has_zp: bool | None = None,
    include_fp_type: bool = True,
    device_capability: int | None = None,
): ...
def check_marlin_supported(
    quant_type: ScalarType,
    group_size: int,
    has_zp: bool = False,
    device_capability: int | None = None,
) -> bool: ...
def verify_marlin_supported(
    quant_type: ScalarType, group_size: int, has_zp: bool = False
) -> None: ...
def verify_marlin_supports_shape(
    output_size_per_partition: int,
    input_size_per_partition: int,
    input_size: int,
    group_size: int,
) -> None: ...
def check_marlin_supports_shape(
    output_size_per_partition: int,
    input_size_per_partition: int,
    input_size: int,
    group_size: int,
) -> tuple[bool, str | None]: ...
def check_marlin_supports_layer(layer: LinearBase, group_size: int) -> bool: ...
def check_moe_marlin_supports_layer(layer: LinearBase, group_size: int) -> bool: ...
def marlin_moe_intermediate_size(w1_packed: torch.Tensor, w2_packed: torch.Tensor): ...
def marlin_make_workspace_new(
    device: torch.device, max_blocks_per_sm: int = 1
) -> torch.Tensor: ...
def marlin_is_k_full(act_order: bool, is_row_parallel: bool) -> bool: ...
def marlin_repeat_scales_on_all_ranks(
    act_order: bool, group_size: int, is_row_parallel: bool
) -> bool: ...
def marlin_make_empty_g_idx(device: torch.device) -> torch.Tensor: ...
def marlin_sort_g_idx(g_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...
def get_scale_perms(): ...
def marlin_permute_scales(
    s: torch.Tensor, size_k: int, size_n: int, group_size: int, is_a_8bit: bool = False
) -> torch.Tensor: ...
def marlin_permute_bias(s: torch.Tensor) -> torch.Tensor: ...
def marlin_act_int8_process_scales(s: torch.Tensor): ...
def marlin_moe_permute_scales(
    s: torch.Tensor, size_k: int, size_n: int, group_size: int, is_a_8bit: bool = False
): ...
def marlin_zero_points(
    zp: torch.Tensor, size_k: int, size_n: int, num_bits: int, is_a_8bit: bool = False
) -> torch.Tensor: ...
def awq_to_marlin_zero_points(
    q_zp_packed: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
    is_a_8bit: bool = False,
) -> torch.Tensor: ...
def moe_awq_to_marlin_zero_points(
    q_zp_packed: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
    is_a_8bit: bool = False,
): ...
def maybe_warn_marlin_atomic_add(device, dtype) -> None: ...
def maybe_warn_marlin_atomic_add_env() -> None: ...
def should_use_atomic_add_reduce(
    m: int, n: int, k: int, device: torch.device, dtype: torch.dtype
) -> bool: ...
def get__quant_fp8_method() -> QuantFP8: ...
def get_marlin_input_dtype(prefix: str | None = None): ...
def marlin_quant_input(x: torch.Tensor, quant_dtype: torch.dtype): ...
def apply_gptq_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_zp: torch.Tensor,
    g_idx: torch.Tensor,
    g_idx_sort_indices: torch.Tensor,
    workspace: torch.Tensor,
    wtype: ScalarType,
    output_size_per_partition: int,
    input_size_per_partition: int,
    is_k_full: bool,
    input_global_scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    use_fp32_reduce: bool = ...,
    input_dtype: torch.dtype | None = None,
) -> torch.Tensor: ...
def apply_awq_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_zp: torch.Tensor,
    g_idx: torch.Tensor,
    g_idx_sort_indices: torch.Tensor,
    workspace: torch.Tensor,
    quant_type: ScalarType,
    output_size_per_partition: int,
    input_size_per_partition: int,
    input_global_scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    use_fp32_reduce: bool = ...,
    input_dtype: torch.dtype | None = None,
) -> torch.Tensor: ...
