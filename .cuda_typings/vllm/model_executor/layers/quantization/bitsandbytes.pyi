import torch
from _typeshed import Incomplete
from typing import Any
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig as FusedMoEConfig,
    FusedMoEQuantConfig as FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE as FusedMoE,
    FusedMoEMethodBase as FusedMoEMethodBase,
)
from vllm.model_executor.layers.linear import (
    LinearBase as LinearBase,
    LinearMethodBase as LinearMethodBase,
    UnquantizedLinearMethod as UnquantizedLinearMethod,
    set_weight_attrs as set_weight_attrs,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
    QuantizationMethods as QuantizationMethods,
)
from vllm.platforms import current_platform as current_platform
from vllm.utils.torch_utils import (
    direct_register_custom_op as direct_register_custom_op,
)

class BitsAndBytesConfig(QuantizationConfig):
    load_in_8bit: Incomplete
    load_in_4bit: Incomplete
    bnb_4bit_compute_dtype: Incomplete
    bnb_4bit_quant_storage: Incomplete
    bnb_4bit_quant_type: Incomplete
    bnb_4bit_use_double_quant: Incomplete
    llm_int8_enable_fp32_cpu_offload: Incomplete
    llm_int8_has_fp16_weight: Incomplete
    llm_int8_skip_modules: Incomplete
    llm_int8_threshold: Incomplete
    def __init__(
        self,
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
        bnb_4bit_compute_dtype: str = "float32",
        bnb_4bit_quant_storage: str = "uint8",
        bnb_4bit_quant_type: str = "fp4",
        bnb_4bit_use_double_quant: bool = False,
        llm_int8_enable_fp32_cpu_offload: bool = False,
        llm_int8_has_fp16_weight: bool = False,
        llm_int8_skip_modules: list[str] | None = None,
        llm_int8_threshold: float = 6.0,
    ) -> None: ...
    @classmethod
    def get_name(self) -> QuantizationMethods: ...
    @classmethod
    def get_supported_act_dtypes(self) -> list[torch.dtype]: ...
    @classmethod
    def get_min_capability(cls) -> int: ...
    @staticmethod
    def get_config_filenames() -> list[str]: ...
    @classmethod
    def from_config(cls, config: dict[str, Any]) -> BitsAndBytesConfig: ...
    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> LinearMethodBase | BitsAndBytesMoEMethod | None: ...

def is_layer_skipped_bnb(prefix: str, llm_int8_skip_modules: list[str]): ...
def calculate_quant_ratio(dtype): ...

class BitsAndBytesLinearMethod(LinearMethodBase):
    quant_config: Incomplete
    def __init__(self, quant_config: BitsAndBytesConfig) -> None: ...
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

apply_bnb_4bit: Incomplete

class BitsAndBytesMoEMethod(FusedMoEMethodBase):
    quant_config: Incomplete
    def __init__(
        self, quant_config: BitsAndBytesConfig, moe: FusedMoEConfig
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
