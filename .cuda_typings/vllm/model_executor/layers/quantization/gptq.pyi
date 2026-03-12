import torch
from _typeshed import Incomplete
from enum import Enum
from typing import Any
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.fused_moe.layer import FusedMoE as FusedMoE
from vllm.model_executor.layers.linear import LinearMethodBase as LinearMethodBase
from vllm.model_executor.layers.quantization import (
    QuantizationMethods as QuantizationMethods,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig as QuantizationConfig,
    QuantizeMethodBase as QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.gptq_utils import (
    get_linear_quant_method as get_linear_quant_method,
)
from vllm.model_executor.models.utils import WeightsMapper as WeightsMapper
from vllm.model_executor.parameter import (
    ChannelQuantScaleParameter as ChannelQuantScaleParameter,
    GroupQuantScaleParameter as GroupQuantScaleParameter,
    PackedColumnParameter as PackedColumnParameter,
    PackedvLLMParameter as PackedvLLMParameter,
    RowvLLMParameter as RowvLLMParameter,
)
from vllm.transformers_utils.config import (
    get_safetensors_params_metadata as get_safetensors_params_metadata,
)
from vllm.utils.collection_utils import is_list_of as is_list_of

logger: Incomplete

class GPTQConfig(QuantizationConfig):
    dynamic: Incomplete
    weight_bits: Incomplete
    group_size: Incomplete
    desc_act: Incomplete
    lm_head_quantized: Incomplete
    pack_factor: Incomplete
    modules_in_block_to_quantize: Incomplete
    autoround_version: Incomplete
    checkpoint_format: Incomplete
    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        desc_act: bool,
        lm_head_quantized: bool,
        dynamic: dict[str, dict[str, int | bool]],
        autoround_version: str = "",
        modules_in_block_to_quantize: list[str] | None = None,
        checkpoint_format: str = "",
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
    def from_config(cls, config: dict[str, Any]) -> GPTQConfig: ...
    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> GPTQLinearMethod | QuantizeMethodBase | None: ...
    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper): ...
    def maybe_update_config(self, model_name: str, revision: str | None = None): ...

class ExllamaState(Enum):
    UNUSED = ...
    UNINITIALIZED = ...
    READY = ...

class GPTQLinearMethod(LinearMethodBase):
    quant_config: Incomplete
    use_v2_format: Incomplete
    def __init__(self, quant_config: GPTQConfig) -> None: ...
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
