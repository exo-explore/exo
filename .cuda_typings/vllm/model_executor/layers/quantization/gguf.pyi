import torch
from _typeshed import Incomplete
from collections.abc import Mapping
from torch.nn.parameter import Parameter, UninitializedParameter
from typing import Any
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation as MoEActivation,
    apply_moe_activation as apply_moe_activation,
)
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
)
from vllm.model_executor.layers.quantization import (
    QuantizationMethods as QuantizationMethods,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig as QuantizationConfig,
    QuantizeMethodBase as QuantizeMethodBase,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod as UnquantizedEmbeddingMethod,
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import WeightsMapper as WeightsMapper
from vllm.model_executor.utils import set_weight_attrs as set_weight_attrs
from vllm.platforms import current_platform as current_platform
from vllm.utils.torch_utils import (
    direct_register_custom_op as direct_register_custom_op,
)

logger: Incomplete

class GGUFConfig(QuantizationConfig):
    unquantized_modules: Incomplete
    def __init__(self, unquantized_modules: list[str] | None = None) -> None: ...
    def get_name(self) -> QuantizationMethods: ...
    def get_supported_act_dtypes(self) -> list[torch.dtype]: ...
    @classmethod
    def get_min_capability(cls) -> int: ...
    @classmethod
    def get_config_filenames(cls) -> list[str]: ...
    @classmethod
    def from_config(cls, config: dict[str, Any]) -> GGUFConfig: ...
    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None: ...
    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper): ...

def is_layer_skipped_gguf(
    prefix: str,
    unquantized_modules: list[str],
    fused_mapping: Mapping[str, list[str]] = ...,
): ...

UNQUANTIZED_TYPES: Incomplete
STANDARD_QUANT_TYPES: Incomplete
KQUANT_TYPES: Incomplete
IMATRIX_QUANT_TYPES: Incomplete
DEQUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES | IMATRIX_QUANT_TYPES
MMVQ_QUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES | IMATRIX_QUANT_TYPES
MMQ_QUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES
fused_mul_mat_gguf: Incomplete
fused_moe_gguf: Incomplete
apply_gguf_embedding: Incomplete

class GGUFLinearMethod(LinearMethodBase):
    quant_config: Incomplete
    def __init__(self, quant_config: GGUFConfig) -> None: ...
    params_dtype: Incomplete
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
    def process_weights_after_loading(self, layer: torch.nn.Module): ...
    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor: ...

class GGUFMoEMethod(FusedMoEMethodBase):
    quant_config: Incomplete
    def __init__(self, quant_config: GGUFConfig, moe: FusedMoEConfig) -> None: ...
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

class GGUFEmbeddingMethod(GGUFLinearMethod):
    def embedding(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor: ...

class GGUFUninitializedParameter(UninitializedParameter):
    cls_to_become = Parameter
    data_container: list[torch.Tensor]
