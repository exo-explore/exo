import torch
from _typeshed import Incomplete
from typing import Any
from vllm._custom_ops import cpu_gemm_wna16 as cpu_gemm_wna16
from vllm.logger import init_logger as init_logger
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
    pack_cols as pack_cols,
    unpack_cols as unpack_cols,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead as ParallelLMHead,
)
from vllm.model_executor.models.utils import WeightsMapper as WeightsMapper
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter as GroupQuantScaleParameter,
    PackedvLLMParameter as PackedvLLMParameter,
)
from vllm.platforms import current_platform as current_platform
from vllm.transformers_utils.config import (
    get_safetensors_params_metadata as get_safetensors_params_metadata,
)

logger: Incomplete

class CPUAWQConfig(QuantizationConfig):
    pack_factor: Incomplete
    group_size: Incomplete
    zero_point: Incomplete
    lm_head_quantized: Incomplete
    weight_bits: Incomplete
    modules_to_not_convert: Incomplete
    full_config: Incomplete
    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        zero_point: bool,
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
    def from_config(cls, config: dict[str, Any]) -> CPUAWQConfig: ...
    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant
    ) -> QuantizationMethods | None: ...
    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None: ...
    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper): ...
    def maybe_update_config(self, model_name: str, revision: str | None = None): ...

class CPUAWQLinearMethod(LinearMethodBase):
    quant_config: Incomplete
    def __init__(self, quant_config: CPUAWQConfig) -> None: ...
    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None: ...
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None: ...
    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor: ...
