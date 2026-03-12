import torch
from _typeshed import Incomplete
from typing import Any
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.attention import Attention as Attention
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
from vllm.model_executor.layers.quantization.kv_cache import (
    BaseKVCacheMethod as BaseKVCacheMethod,
)
from vllm.model_executor.layers.quantization.utils.petit_utils import (
    apply_petit_nvfp4_linear as apply_petit_nvfp4_linear,
    prepare_nvfp4_layer_for_petit as prepare_nvfp4_layer_for_petit,
    verify_petit_nvfp4_supported as verify_petit_nvfp4_supported,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped as is_layer_skipped,
)
from vllm.model_executor.parameter import (
    ModelWeightParameter as ModelWeightParameter,
    PerTensorScaleParameter as PerTensorScaleParameter,
)
from vllm.platforms import current_platform as current_platform

logger: Incomplete

class PetitNvFp4Config(QuantizationConfig):
    is_checkpoint_nvfp4_serialized: Incomplete
    group_size: Incomplete
    kv_cache_quant_algo: Incomplete
    exclude_modules: Incomplete
    def __init__(
        self,
        is_checkpoint_nvfp4_serialized: bool = False,
        kv_cache_quant_algo: str | None = None,
        group_size: int | None = None,
        exclude_modules: list[str] | None = None,
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
    def from_config(cls, config: dict[str, Any]) -> PetitNvFp4Config: ...
    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant
    ) -> QuantizationMethods | None: ...
    @classmethod
    def is_petit_nvfp4_compatible(cls, quant_config: dict[str, Any]) -> bool: ...
    def is_layer_excluded(self, prefix: str, exclude_modules: list[str]) -> bool: ...
    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None: ...
    def get_scaled_act_names(self) -> list[str]: ...
    def require_group_size(self) -> int: ...
    def require_kv_cache_quant_algo(self) -> str: ...
    def require_exclude_modules(self) -> list[str]: ...

class PetitFp8KVCacheMethod(BaseKVCacheMethod):
    def __init__(self, quant_config: PetitNvFp4Config) -> None: ...

class PetitNvFp4LinearMethod(LinearMethodBase):
    quant_config: Incomplete
    def __init__(self, quant_config: PetitNvFp4Config) -> None: ...
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
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None: ...
    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor: ...
