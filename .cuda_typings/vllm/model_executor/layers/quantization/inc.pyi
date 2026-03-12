import torch
from _typeshed import Incomplete
from typing import Any
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.linear import (
    LinearBase as LinearBase,
    UnquantizedLinearMethod as UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
    QuantizationMethods as QuantizationMethods,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead as ParallelLMHead,
)
from vllm.model_executor.models.utils import WeightsMapper as WeightsMapper
from vllm.platforms import current_platform as current_platform
from vllm.scalar_type import scalar_types as scalar_types

logger: Incomplete

class INCConfig(QuantizationConfig):
    SUPPORTED_BITS: Incomplete
    SUPPORTED_DTYPES: Incomplete
    SUPPORTED_FORMATS: Incomplete
    SUPPORTED_BACKENDS: Incomplete
    weight_bits: Incomplete
    group_size: Incomplete
    sym: Incomplete
    packing_format: Incomplete
    block_name_to_quantize: Incomplete
    extra_config: Incomplete
    data_type: Incomplete
    backend: Incomplete
    pack_factor: Incomplete
    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        sym: bool = True,
        packing_format: str = "auto_round:auto_gptq",
        block_name_to_quantize: str | list[str] | None = None,
        extra_config: dict[str, Any] | None = None,
        data_type: str = "int",
        backend: str = "auto",
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
    def from_config(cls, config: dict[str, Any]) -> INCConfig: ...
    def get_layer_config(self, layer, layer_name: str): ...
    def check_quantized(self, weight_bits: int) -> bool: ...
    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper): ...
    def apply_awq_quant_layer(self, layer, prefix: str, backend: str = "auto"): ...
    def apply_gptq_quant_layer(self, layer, prefix: str, backend: str = "auto"): ...
    def apply_ipex_quant_layer(self, layer, prefix: str): ...
    def get_quant_method(self, layer: torch.nn.Module, prefix: str): ...
    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant
    ) -> QuantizationMethods | None: ...
