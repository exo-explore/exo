import functools
import torch
from _typeshed import Incomplete
from collections.abc import Callable as Callable, Sequence
from typing import Any
from vllm._aiter_ops import rocm_aiter_ops as rocm_aiter_ops
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8 as QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape as GroupShape,
    get_fp8_min_max as get_fp8_min_max,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    CUTLASS_BLOCK_FP8_SUPPORTED as CUTLASS_BLOCK_FP8_SUPPORTED,
    all_close_1d as all_close_1d,
    per_tensor_dequantize as per_tensor_dequantize,
)
from vllm.model_executor.parameter import (
    BlockQuantScaleParameter as BlockQuantScaleParameter,
    ChannelQuantScaleParameter as ChannelQuantScaleParameter,
    PerTensorScaleParameter as PerTensorScaleParameter,
)
from vllm.model_executor.utils import (
    replace_parameter as replace_parameter,
    set_weight_attrs as set_weight_attrs,
)
from vllm.platforms import current_platform as current_platform
from vllm.triton_utils import tl as tl, triton as triton
from vllm.utils.deep_gemm import (
    fp8_gemm_nt as fp8_gemm_nt,
    get_tma_aligned_size as get_tma_aligned_size,
    is_deep_gemm_e8m0_used as is_deep_gemm_e8m0_used,
    is_deep_gemm_supported as is_deep_gemm_supported,
    should_use_deepgemm_for_fp8_linear as should_use_deepgemm_for_fp8_linear,
    transform_sf_into_required_layout as transform_sf_into_required_layout,
)
from vllm.utils.flashinfer import (
    flashinfer_fp8_blockscale_gemm as flashinfer_fp8_blockscale_gemm,
    is_flashinfer_fp8_blockscale_gemm_supported as is_flashinfer_fp8_blockscale_gemm_supported,
    should_use_flashinfer_for_blockscale_fp8_gemm as should_use_flashinfer_for_blockscale_fp8_gemm,
)
from vllm.utils.torch_utils import (
    direct_register_custom_op as direct_register_custom_op,
)

logger: Incomplete

def is_fp8(x: torch.dtype | torch.Tensor) -> bool: ...
def cutlass_scaled_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = ...,
) -> torch.Tensor: ...

class W8A8BlockFp8LinearOp:
    weight_group_shape: Incomplete
    act_quant_group_shape: Incomplete
    is_deep_gemm_supported: Incomplete
    is_hopper: Incomplete
    use_deep_gemm_e8m0: Incomplete
    is_flashinfer_supported: Incomplete
    deepgemm_input_quant_op: Incomplete
    def __init__(
        self,
        weight_group_shape: GroupShape,
        act_quant_group_shape: GroupShape,
        cutlass_block_fp8_supported: bool = ...,
        use_aiter_and_is_supported: bool = False,
    ) -> None: ...
    def apply(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        input_scale: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

def input_to_float8(
    x: torch.Tensor, dtype: torch.dtype | None = None
) -> tuple[torch.Tensor, torch.Tensor]: ...
def silu_mul_per_token_group_quant_fp8_colmajor(
    input: torch.Tensor,
    output: torch.Tensor | None = None,
    use_ue8m0: bool | None = None,
    eps: float = 1e-10,
): ...
def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype | None = None,
    column_major_scales: bool = False,
    tma_aligned_scales: bool = False,
    out_q: torch.Tensor | None = None,
    use_ue8m0: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def per_token_group_quant_fp8_packed_for_deepgemm(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    use_ue8m0: bool | None = None,
    out_q: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]: ...
@functools.lru_cache
def get_w8a8_block_fp8_configs(
    N: int, K: int, block_n: int, block_k: int
) -> dict[int, Any] | None: ...
def w8a8_triton_block_scaled_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = ...,
) -> torch.Tensor: ...
def requant_weight_ue8m0_inplace(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: Sequence[int] = (128, 128),
) -> None: ...
def deepgemm_post_process_fp8_weight_block(
    wq: torch.Tensor, ws: torch.Tensor, quant_block_shape: tuple[int], use_e8m0: bool
) -> tuple[torch.Tensor, torch.Tensor]: ...
def prepare_fp8_moe_layer_for_deepgemm(
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    block_shape: tuple[int],
): ...
def validate_fp8_block_shape(
    layer: torch.nn.Module,
    input_size: int,
    output_size: int,
    input_size_per_partition: int,
    output_partition_sizes: list[int],
    block_size: list[int],
) -> None: ...
def create_fp8_weight_parameter(
    output_size_per_partition: int,
    input_size_per_partition: int,
    weight_loader: Callable | None,
) -> torch.nn.Parameter: ...
def create_fp8_scale_parameter(
    parameter_type: torch.nn.Parameter,
    output_partition_sizes: list[int],
    input_size_per_partition: int,
    block_size: list[int] | None,
    weight_loader: Callable | None,
) -> torch.nn.Parameter: ...
def create_fp8_input_scale(
    output_partition_sizes: list[int], weight_loader: Callable | None
) -> torch.nn.Parameter: ...
def process_fp8_weight_tensor_strategy(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    logical_widths: list[int],
    input_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]: ...
def process_fp8_weight_channel_strategy(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]: ...
def process_fp8_weight_block_strategy(
    weight: torch.Tensor, weight_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]: ...
def maybe_post_process_fp8_weight_block(layer: torch.nn.Module): ...
def process_fp8_weight_tensor_strategy_moe(
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    shard_size: int,
    num_experts: int,
    is_act_and_mul: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def process_fp8_input_tensor_strategy_moe(
    w13_input_scale: torch.Tensor, w2_input_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]: ...
