import torch
from _typeshed import Incomplete
from typing import Any
from vllm.logger import init_logger as init_logger
from vllm.model_executor.kernels.linear import (
    MPLinearLayerConfig as MPLinearLayerConfig,
    choose_mp_linear_kernel as choose_mp_linear_kernel,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig as FusedMoEConfig,
    FusedMoEQuantConfig as FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
    fused_marlin_moe as fused_marlin_moe,
)
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE as FusedMoE,
    FusedMoEMethodBase as FusedMoEMethodBase,
    FusedMoeWeightScaleSupported as FusedMoeWeightScaleSupported,
    UnquantizedFusedMoEMethod as UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import (
    LinearMethodBase as LinearMethodBase,
    set_weight_attrs as set_weight_attrs,
)
from vllm.model_executor.layers.quantization import (
    QuantizationMethods as QuantizationMethods,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig as QuantizationConfig,
    QuantizeMethodBase as QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils import (
    replace_parameter as replace_parameter,
)
from vllm.model_executor.layers.quantization.utils.gptq_utils import (
    get_dynamic_override as get_dynamic_override,
    get_linear_quant_method as get_linear_quant_method,
    override_config as override_config,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    check_marlin_supported as check_marlin_supported,
    check_moe_marlin_supports_layer as check_moe_marlin_supports_layer,
    get_marlin_input_dtype as get_marlin_input_dtype,
    marlin_act_int8_process_scales as marlin_act_int8_process_scales,
    marlin_make_workspace_new as marlin_make_workspace_new,
    marlin_moe_permute_scales as marlin_moe_permute_scales,
    marlin_permute_bias as marlin_permute_bias,
    marlin_repeat_scales_on_all_ranks as marlin_repeat_scales_on_all_ranks,
    verify_marlin_supported as verify_marlin_supported,
)
from vllm.model_executor.parameter import (
    ChannelQuantScaleParameter as ChannelQuantScaleParameter,
    GroupQuantScaleParameter as GroupQuantScaleParameter,
    PackedColumnParameter as PackedColumnParameter,
    PackedvLLMParameter as PackedvLLMParameter,
    RowvLLMParameter as RowvLLMParameter,
)
from vllm.platforms import current_platform as current_platform
from vllm.scalar_type import scalar_types as scalar_types
from vllm.transformers_utils.config import (
    get_safetensors_params_metadata as get_safetensors_params_metadata,
)
from vllm.utils.collection_utils import is_list_of as is_list_of

logger: Incomplete

def get_moe_quant_method(
    config: GPTQMarlinConfig, layer: torch.nn.Module, prefix: str, moe_method_cls: type
): ...

class GPTQMarlinConfig(QuantizationConfig):
    TYPE_MAP: Incomplete
    dynamic: Incomplete
    weight_bits: Incomplete
    is_sym: Incomplete
    pack_factor: Incomplete
    group_size: Incomplete
    desc_act: Incomplete
    lm_head_quantized: Incomplete
    full_config: Incomplete
    quant_type: Incomplete
    modules_in_block_to_quantize: Incomplete
    autoround_version: Incomplete
    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        desc_act: bool,
        is_sym: bool,
        lm_head_quantized: bool,
        dynamic: dict[str, dict[str, int | bool]],
        full_config: dict[str, Any],
        modules_in_block_to_quantize: list[str] | None = None,
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
    def from_config(cls, config: dict[str, Any]) -> GPTQMarlinConfig: ...
    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant
    ) -> QuantizationMethods | None: ...
    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None: ...
    @classmethod
    def is_gptq_marlin_compatible(cls, quant_config: dict[str, Any]): ...
    def apply_vllm_mapper(self, hf_to_vllm_mapper) -> None: ...
    def maybe_update_config(self, model_name: str, revision: str | None = None): ...

class GPTQMarlinLinearMethod(LinearMethodBase):
    quant_config: Incomplete
    input_dtype: Incomplete
    quant_type: Incomplete
    def __init__(self, quant_config: GPTQMarlinConfig) -> None: ...
    kernel: Incomplete
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

class GPTQMarlinMoEMethod(FusedMoEMethodBase):
    quant_config: Incomplete
    quant_type: Incomplete
    input_dtype: Incomplete
    use_marlin: bool
    def __init__(self, quant_config: GPTQMarlinConfig, moe: FusedMoEConfig) -> None: ...
    is_k_full: Incomplete
    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ): ...
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None: ...
    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None: ...
    def select_gemm_impl(self, prepare_finalize, layer: torch.nn.Module): ...
    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...
