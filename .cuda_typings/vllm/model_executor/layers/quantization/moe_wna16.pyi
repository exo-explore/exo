import torch
from _typeshed import Incomplete
from typing import Any
from vllm.distributed import (
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
    get_tp_group as get_tp_group,
)
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation as MoEActivation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig as FusedMoEQuantConfig,
    int4_w4a16_moe_quant_config as int4_w4a16_moe_quant_config,
    int8_w8a16_moe_quant_config as int8_w8a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE as FusedMoE,
    FusedMoEConfig as FusedMoEConfig,
    FusedMoEMethodBase as FusedMoEMethodBase,
    FusedMoeWeightScaleSupported as FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod as UnquantizedFusedMoEMethod,
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
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    check_marlin_supports_layer as check_marlin_supports_layer,
)
from vllm.model_executor.utils import set_weight_attrs as set_weight_attrs
from vllm.platforms import current_platform as current_platform

class MoeWNA16Config(QuantizationConfig):
    weight_bits: Incomplete
    group_size: Incomplete
    has_zp: Incomplete
    bit8_pack_factor: Incomplete
    lm_head_quantized: Incomplete
    linear_quant_method: Incomplete
    full_config: Incomplete
    use_marlin: bool
    modules_to_not_convert: Incomplete
    def __init__(
        self,
        linear_quant_method: str,
        weight_bits: int,
        group_size: int,
        has_zp: bool,
        lm_head_quantized: bool,
        modules_to_not_convert: list[str] | None,
        full_config: dict[str, Any],
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
    def from_config(cls, config: dict[str, Any]) -> MoeWNA16Config: ...
    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant
    ) -> QuantizationMethods | None: ...
    @classmethod
    def is_moe_wna16_compatible(cls, quant_config: dict[str, Any]): ...
    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None: ...

def is_layer_skipped_quant(prefix: str, modules_to_not_convert: list[str]): ...

class MoeWNA16Method(FusedMoEMethodBase):
    quant_config: Incomplete
    def __init__(self, quant_config: MoeWNA16Config, moe: FusedMoEConfig) -> None: ...
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
    def get_weight_loader(layer, weight_loader): ...
