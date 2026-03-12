import torch
from _typeshed import Incomplete
from typing import Any
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.quark.schemes import QuarkScheme
from vllm.model_executor.models.utils import WeightsMapper

__all__ = ["QuarkLinearMethod"]

class QuarkConfig(QuantizationConfig):
    quant_config: Incomplete
    kv_cache_group: Incomplete
    kv_cache_config: Incomplete
    pack_method: Incomplete
    dynamic_mxfp4_quant: bool
    def __init__(
        self,
        quant_config: dict[str, Any],
        kv_cache_group: list[str] | None = None,
        kv_cache_config: dict[str, Any] | None = None,
        pack_method: str = "reorder",
    ) -> None: ...
    hf_config: Incomplete
    def maybe_update_config(self, model_name: str, revision: str | None = None): ...
    def get_linear_method(self) -> QuarkLinearMethod: ...
    def get_supported_act_dtypes(cls) -> list[torch.dtype]: ...
    @classmethod
    def get_min_capability(cls) -> int: ...
    def get_name(self) -> QuantizationMethods: ...
    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper): ...
    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None: ...
    @classmethod
    def from_config(cls, config: dict[str, Any]) -> QuarkConfig: ...
    @classmethod
    def get_config_filenames(cls) -> list[str]: ...
    def is_mxfp4_quant(self, prefix: str, layer: torch.nn.Module) -> bool: ...
    def get_scheme(
        self, layer: torch.nn.Module, layer_name: str, dynamic_mxfp4_quant: bool = False
    ) -> QuarkScheme: ...
    def get_cache_scale(self, name: str) -> str | None: ...

class QuarkLinearMethod(LinearMethodBase):
    quantization_config: Incomplete
    def __init__(self, quantization_config: QuarkConfig) -> None: ...
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None: ...
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
    ): ...

class QuarkKVCacheMethod(BaseKVCacheMethod):
    def __init__(self, quant_config: QuarkConfig) -> None: ...
    @staticmethod
    def validate_kv_cache_config(kv_cache_config: dict[str, Any] | None): ...
