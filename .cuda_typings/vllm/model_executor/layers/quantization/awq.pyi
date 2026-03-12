import torch
from _typeshed import Incomplete
from typing import Any
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.fused_moe.layer import FusedMoE as FusedMoE
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
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped as is_layer_skipped,
)
from vllm.model_executor.models.utils import WeightsMapper as WeightsMapper
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter as GroupQuantScaleParameter,
    PackedvLLMParameter as PackedvLLMParameter,
)
from vllm.transformers_utils.config import (
    get_safetensors_params_metadata as get_safetensors_params_metadata,
)

logger: Incomplete

class AWQConfig(QuantizationConfig):
    weight_bits: Incomplete
    group_size: Incomplete
    zero_point: Incomplete
    modules_to_not_convert: Incomplete
    pack_factor: Incomplete
    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        zero_point: bool,
        modules_to_not_convert: list[str] | None = None,
    ) -> None: ...
    def get_name(self) -> QuantizationMethods: ...
    def get_supported_act_dtypes(self) -> list[torch.dtype]: ...
    @classmethod
    def get_min_capability(cls) -> int: ...
    @staticmethod
    def get_config_filenames() -> list[str]: ...
    @classmethod
    def from_config(cls, config: dict[str, Any]) -> AWQConfig: ...
    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> LinearMethodBase | QuantizeMethodBase | None: ...
    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper): ...
    def maybe_update_config(self, model_name: str, revision: str | None = None): ...

class AWQLinearMethod(LinearMethodBase):
    quant_config: Incomplete
    def __init__(self, quant_config: AWQConfig) -> None: ...
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
