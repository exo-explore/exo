import torch
from _typeshed import Incomplete
from typing import Any
from vllm.model_executor.kernels.linear import (
    init_fp8_linear_kernel as init_fp8_linear_kernel,
)
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.model_executor.layers.linear import (
    LinearBase as LinearBase,
    UnquantizedLinearMethod as UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import (
    QuantizationMethods as QuantizationMethods,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase as QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.fp8 import (
    Fp8Config as Fp8Config,
    Fp8KVCacheMethod as Fp8KVCacheMethod,
    Fp8LinearMethod as Fp8LinearMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped as is_layer_skipped,
    kFp8DynamicTokenSym as kFp8DynamicTokenSym,
)
from vllm.platforms import current_platform as current_platform

class PTPCFp8Config(Fp8Config):
    def __init__(
        self,
        activation_scheme: str = "dynamic",
        ignored_layers: list[str] | None = None,
    ) -> None: ...
    @classmethod
    def get_name(cls) -> QuantizationMethods: ...
    @classmethod
    def from_config(cls, config: dict[str, Any]) -> PTPCFp8Config: ...
    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None: ...

class PTPCFp8LinearMethod(Fp8LinearMethod):
    fp8_linear: Incomplete
    def __init__(self, quant_config: PTPCFp8Config) -> None: ...
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None: ...
    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor: ...
