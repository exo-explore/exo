import torch
from _typeshed import Incomplete
from typing import Any
from vllm.distributed import (
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
    get_tp_group as get_tp_group,
)
from vllm.model_executor.layers.fused_moe import (
    FusedMoE as FusedMoE,
    FusedMoEConfig as FusedMoEConfig,
    FusedMoEMethodBase as FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig as FusedMoEQuantConfig,
    int8_w8a16_moe_quant_config as int8_w8a16_moe_quant_config,
)
from vllm.model_executor.layers.linear import (
    LinearBase as LinearBase,
    UnquantizedLinearMethod as UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import (
    QuantizationMethods as QuantizationMethods,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig as QuantizationConfig,
    QuantizeMethodBase as QuantizeMethodBase,
)
from vllm.model_executor.utils import set_weight_attrs as set_weight_attrs

class ExpertsInt8Config(QuantizationConfig):
    def __init__(self) -> None: ...
    @classmethod
    def get_name(cls) -> QuantizationMethods: ...
    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]: ...
    @classmethod
    def get_min_capability(cls) -> int: ...
    @classmethod
    def get_config_filenames(cls) -> list[str]: ...
    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ExpertsInt8Config: ...
    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None: ...

class ExpertsInt8MoEMethod(FusedMoEMethodBase):
    quant_config: Incomplete
    def __init__(
        self, quant_config: ExpertsInt8Config, moe: FusedMoEConfig
    ) -> None: ...
    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ): ...
    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None: ...
    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...
    @staticmethod
    def quantizing_weight_loader(layer, weight_loader): ...

def quantize_in_place_and_get_scales(weight: torch.Tensor) -> torch.Tensor: ...
