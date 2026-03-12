import torch
from _typeshed import Incomplete
from typing import Any
from vllm._custom_ops import (
    cutlass_scaled_fp4_mm as cutlass_scaled_fp4_mm,
    fusedQuantizeMx as fusedQuantizeMx,
    fusedQuantizeNv as fusedQuantizeNv,
    matmul_mxf4_bf16_tn as matmul_mxf4_bf16_tn,
)
from vllm.model_executor.layers.linear import (
    LinearBase as LinearBase,
    LinearMethodBase as LinearMethodBase,
    UnquantizedLinearMethod as UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import (
    QuantizationMethods as QuantizationMethods,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.layers.quantization.qutlass_utils import (
    to_blocked as to_blocked,
)
from vllm.model_executor.utils import set_weight_attrs as set_weight_attrs
from vllm.platforms import current_platform as current_platform
from vllm.utils.torch_utils import (
    direct_register_custom_op as direct_register_custom_op,
)

class FPQuantConfig(QuantizationConfig):
    hadamard_group_size: Incomplete
    forward_dtype: Incomplete
    forward_method: Incomplete
    pseudoquantization: Incomplete
    modules_to_not_convert: Incomplete
    def __init__(
        self,
        hadamard_group_size: int = 32,
        forward_dtype: str = "mxfp4",
        forward_method: str = "abs_max",
        pseudoquantization: bool = False,
        modules_to_not_convert: list[str] | None = None,
    ) -> None: ...
    @classmethod
    def get_name(cls) -> QuantizationMethods: ...
    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]: ...
    @classmethod
    def get_min_capability(cls) -> int: ...
    @classmethod
    def get_config_filenames(cls) -> list[str]: ...
    @classmethod
    def from_config(cls, config: dict[str, Any]) -> FPQuantConfig: ...
    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> LinearMethodBase | None: ...

class FPQuantLinearMethod(LinearMethodBase):
    quant_config: Incomplete
    def __init__(self, quant_config: FPQuantConfig) -> None: ...
    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ): ...
    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor: ...

def fused_quantize_mx(
    x_flat: torch.Tensor, hadamard_matrix: torch.Tensor, forward_method: str
) -> tuple[torch.Tensor, torch.Tensor]: ...
def fused_quantize_mx_fake(x_flat, hadamard_matrix, forward_method): ...
def matmul_mxf4_bf16(
    x: torch.Tensor,
    w: torch.Tensor,
    xs: torch.Tensor,
    ws: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor: ...
def matmul_mxf4_bf16_fake(x, w, xs, ws, alpha): ...
def fused_quantize_nv(
    x_flat: torch.Tensor, hadamard_matrix: torch.Tensor, global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]: ...
def fused_quantize_nv_fake(x_flat, hadamard_matrix, global_scale): ...
def matmul_nvf4_bf16(
    x: torch.Tensor,
    w: torch.Tensor,
    xs: torch.Tensor,
    ws: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor: ...
def matmul_nvf4_bf16_fake(x, w, xs, ws, alpha): ...
def quantized_forward(
    x: torch.Tensor,
    qweight: torch.Tensor,
    weight_scales: torch.Tensor,
    weight_global_scale: torch.Tensor,
    act_global_scale: torch.Tensor,
    bias: torch.Tensor | None,
    forward_hadamard_matrix: torch.Tensor,
    forward_method: str,
    forward_dtype: str,
) -> torch.Tensor: ...
