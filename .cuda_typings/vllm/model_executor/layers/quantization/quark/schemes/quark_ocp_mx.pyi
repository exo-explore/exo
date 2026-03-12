import torch
from .quark_scheme import QuarkScheme as QuarkScheme
from _typeshed import Incomplete
from collections.abc import Callable as Callable
from fractions import Fraction
from functools import cache
from typing import Any
from vllm import envs as envs
from vllm._aiter_ops import rocm_aiter_ops as rocm_aiter_ops
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    dequant_mxfp4 as dequant_mxfp4,
    quant_dequant_mxfp4 as quant_dequant_mxfp4,
)
from vllm.model_executor.layers.quantization.utils.mxfp6_utils import (
    dequant_mxfp6 as dequant_mxfp6,
    quant_dequant_mxfp6 as quant_dequant_mxfp6,
)
from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import (
    OCP_MX_BLOCK_SIZE as OCP_MX_BLOCK_SIZE,
    OCP_MX_Scheme as OCP_MX_Scheme,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter as GroupQuantScaleParameter,
    ModelWeightParameter as ModelWeightParameter,
    PackedvLLMParameter as PackedvLLMParameter,
)
from vllm.model_executor.utils import set_weight_attrs as set_weight_attrs
from vllm.platforms import current_platform as current_platform
from vllm.utils.torch_utils import (
    direct_register_custom_op as direct_register_custom_op,
)

logger: Incomplete

@cache
def is_rocm_aiter_fp4_asm_gemm_enabled() -> bool: ...
def gemm_with_dynamic_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    rocm_use_aiter_fp4_asm_gemm: bool = False,
    out_dtype: torch.dtype | None = ...,
    x_scales: torch.Tensor | None = None,
) -> torch.Tensor: ...
def gemm_with_dynamic_quant_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    x_scales: torch.Tensor = None,
    rocm_use_aiter_fp4_asm_gemm: bool = False,
    out_dtype: torch.dtype | None = ...,
) -> torch.Tensor: ...

class QuarkOCP_MX(QuarkScheme):
    out_dtype: Incomplete
    qscheme: str
    weight_quant_spec: Incomplete
    input_quant_spec: Incomplete
    dynamic_mxfp4_quant: Incomplete
    weight_dtype: Incomplete
    input_dtype: Incomplete
    ocp_mx_scheme: Incomplete
    packed_factor: int | Fraction
    dequant_func: Incomplete
    quant_dequant_func: Incomplete
    static_input_scales: Incomplete
    emulate: Incomplete
    rocm_use_aiter_fp4_asm_gemm: Incomplete
    def __init__(
        self,
        weight_quant_spec: dict[str, Any],
        input_quant_spec: dict[str, Any],
        dynamic_mxfp4_quant: bool = False,
    ) -> None: ...
    def get_packed_dim(self, dim: int, quant_dtype: str): ...
    @classmethod
    def get_min_capability(cls) -> int: ...
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None: ...
    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ): ...
    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor: ...
