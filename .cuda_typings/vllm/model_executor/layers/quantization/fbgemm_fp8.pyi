import torch
from _typeshed import Incomplete
from torch.nn import Module as Module
from typing import Any
from vllm.logger import init_logger as init_logger
from vllm.model_executor.kernels.linear import (
    init_fp8_linear_kernel as init_fp8_linear_kernel,
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
    QuantizeMethodBase as QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear as apply_fp8_marlin_linear,
    prepare_fp8_layer_for_marlin as prepare_fp8_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped as is_layer_skipped,
    kFp8DynamicTokenSym as kFp8DynamicTokenSym,
    kFp8StaticTokenSym as kFp8StaticTokenSym,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    normalize_e4m3fn_to_e4m3fnuz as normalize_e4m3fn_to_e4m3fnuz,
)
from vllm.model_executor.parameter import (
    ChannelQuantScaleParameter as ChannelQuantScaleParameter,
    ModelWeightParameter as ModelWeightParameter,
)
from vllm.platforms import current_platform as current_platform

logger: Incomplete

class FBGEMMFp8Config(QuantizationConfig):
    ignore_list: Incomplete
    input_scale_ub: Incomplete
    use_marlin: Incomplete
    def __init__(self, ignore_list: list[str], input_scale_ub: float) -> None: ...
    @classmethod
    def get_name(cls) -> QuantizationMethods: ...
    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]: ...
    @classmethod
    def get_min_capability(cls) -> int: ...
    @classmethod
    def get_config_filenames(cls) -> list[str]: ...
    @classmethod
    def from_config(cls, config: dict[str, Any]) -> FBGEMMFp8Config: ...
    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None: ...

class FBGEMMFp8LinearMethod(LinearMethodBase):
    quant_config: Incomplete
    out_dtype: Incomplete
    fp8_linear: Incomplete
    def __init__(self, quant_config: FBGEMMFp8Config) -> None: ...
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
    def process_weights_after_loading(self, layer: Module) -> None: ...
    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor: ...
