import torch
from _typeshed import Incomplete
from compressed_tensors.config import SparsityCompressionConfig
from compressed_tensors.quantization import QuantizationArgs
from typing import Any
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.models.utils import WeightsMapper

__all__ = ["CompressedTensorsLinearMethod"]

QUANTIZATION_SCHEME_MAP_TYPE = dict[str, dict[str, QuantizationArgs] | None]

class CompressedTensorsConfig(QuantizationConfig):
    ignore: Incomplete
    quant_format: Incomplete
    target_scheme_map: Incomplete
    kv_cache_scheme: Incomplete
    sparsity_scheme_map: Incomplete
    sparsity_ignore_list: Incomplete
    config: Incomplete
    total_num_heads: Incomplete
    total_num_kv_heads: Incomplete
    transform_config: Incomplete
    def __init__(
        self,
        target_scheme_map: dict[str, Any],
        ignore: list[str],
        quant_format: str,
        sparsity_scheme_map: dict[str, SparsityCompressionConfig],
        sparsity_ignore_list: list[str],
        kv_cache_scheme: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        transform_config: dict[str, Any] | None = None,
        total_num_heads: int | None = None,
        total_num_kv_heads: int | None = None,
    ) -> None: ...
    def get_linear_method(self) -> CompressedTensorsLinearMethod: ...
    def get_supported_act_dtypes(cls) -> list[torch.dtype]: ...
    @classmethod
    def get_min_capability(cls) -> int: ...
    def get_name(self) -> QuantizationMethods: ...
    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper): ...
    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None: ...
    @classmethod
    def from_config(cls, config: dict[str, Any]) -> CompressedTensorsConfig: ...
    @classmethod
    def get_config_filenames(cls) -> list[str]: ...
    def get_scheme(
        self, layer: torch.nn.Module, layer_name: str | None = None
    ) -> CompressedTensorsScheme | None: ...
    def get_scheme_dict(
        self, layer: torch.nn.Module, layer_name: str | None = None
    ) -> dict[str, QuantizationArgs | str | None] | None: ...
    def has_blocked_weights(self) -> bool: ...
    @staticmethod
    def supports_cutlass_24(
        weight_quant: QuantizationArgs | None,
        input_quant: QuantizationArgs | None,
        sparsity_scheme: SparsityCompressionConfig | None = None,
    ) -> bool: ...

class CompressedTensorsLinearMethod(LinearMethodBase):
    quantization_config: Incomplete
    def __init__(self, quantization_config: CompressedTensorsConfig) -> None: ...
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

class CompressedTensorsKVCacheMethod(BaseKVCacheMethod):
    def __init__(self, quant_config: CompressedTensorsConfig) -> None: ...
    @staticmethod
    def validate_kv_cache_scheme(kv_cache_scheme: dict[str, Any] | None): ...
    def create_weights(self, layer: torch.nn.Module): ...
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None: ...
