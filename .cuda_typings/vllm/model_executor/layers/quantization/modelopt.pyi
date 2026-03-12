import abc
import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from _typeshed import Incomplete
from typing import Any
from vllm.logger import init_logger as init_logger
from vllm.model_executor.kernels.linear import (
    init_fp8_linear_kernel as init_fp8_linear_kernel,
)
from vllm.model_executor.layers.attention import (
    Attention as Attention,
    MLAAttention as MLAAttention,
)
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation as MoEActivation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig as FusedMoEConfig,
    FusedMoEQuantConfig as FusedMoEQuantConfig,
    RoutingMethodType as RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase as FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE as FusedMoE,
    FusedMoeWeightScaleSupported as FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    convert_to_fp8_moe_kernel_format as convert_to_fp8_moe_kernel_format,
    make_fp8_moe_kernel as make_fp8_moe_kernel,
    make_fp8_moe_quant_config as make_fp8_moe_quant_config,
    select_fp8_moe_backend as select_fp8_moe_backend,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp8 import (
    MxFp8MoeBackend as MxFp8MoeBackend,
    select_mxfp8_moe_backend as select_mxfp8_moe_backend,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    convert_to_nvfp4_moe_kernel_format as convert_to_nvfp4_moe_kernel_format,
    is_global_sf_supported_for_nvfp4_backend as is_global_sf_supported_for_nvfp4_backend,
    make_nvfp4_moe_kernel as make_nvfp4_moe_kernel,
    make_nvfp4_moe_quant_config as make_nvfp4_moe_quant_config,
    select_nvfp4_moe_backend as select_nvfp4_moe_backend,
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
from vllm.model_executor.layers.quantization.kv_cache import (
    BaseKVCacheMethod as BaseKVCacheMethod,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    swap_w13_to_w31 as swap_w13_to_w31,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    W8A8BlockFp8LinearOp as W8A8BlockFp8LinearOp,
    process_fp8_input_tensor_strategy_moe as process_fp8_input_tensor_strategy_moe,
    process_fp8_weight_tensor_strategy_moe as process_fp8_weight_tensor_strategy_moe,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    get_marlin_input_dtype as get_marlin_input_dtype,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE as MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_DTYPE as MXFP8_SCALE_DTYPE,
    MXFP8_VALUE_DTYPE as MXFP8_VALUE_DTYPE,
    Mxfp8LinearBackend as Mxfp8LinearBackend,
    Mxfp8LinearOp as Mxfp8LinearOp,
    mxfp8_e4m3_quantize as mxfp8_e4m3_quantize,
    swizzle_mxfp8_scale as swizzle_mxfp8_scale,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    apply_nvfp4_linear as apply_nvfp4_linear,
    convert_to_nvfp4_linear_kernel_format as convert_to_nvfp4_linear_kernel_format,
    select_nvfp4_linear_backend as select_nvfp4_linear_backend,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape as GroupShape,
    is_layer_skipped as is_layer_skipped,
    kFp8DynamicTokenSym as kFp8DynamicTokenSym,
    kFp8StaticTensorSym as kFp8StaticTensorSym,
    kFp8StaticTokenSym as kFp8StaticTokenSym,
    kNvfp4Dynamic as kNvfp4Dynamic,
    kNvfp4Static as kNvfp4Static,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    cutlass_block_fp8_supported as cutlass_block_fp8_supported,
    requantize_with_max_scale as requantize_with_max_scale,
)
from vllm.model_executor.models.utils import WeightsMapper as WeightsMapper
from vllm.model_executor.parameter import (
    BlockQuantScaleParameter as BlockQuantScaleParameter,
    ChannelQuantScaleParameter as ChannelQuantScaleParameter,
    ModelWeightParameter as ModelWeightParameter,
    PerTensorScaleParameter as PerTensorScaleParameter,
)
from vllm.model_executor.utils import (
    replace_parameter as replace_parameter,
    set_weight_attrs as set_weight_attrs,
)
from vllm.utils.flashinfer import (
    flashinfer_trtllm_fp8_block_scale_moe as flashinfer_trtllm_fp8_block_scale_moe,
)

logger: Incomplete
QUANT_ALGOS: Incomplete
KV_CACHE_QUANT_ALGOS: Incomplete

class ModelOptFp8KVCacheMethod(BaseKVCacheMethod):
    def __init__(self, quant_config: ModelOptQuantConfigBase) -> None: ...

class ModelOptQuantConfigBase(QuantizationConfig, metaclass=abc.ABCMeta):
    LinearMethodCls: type
    FusedMoEMethodCls: type
    KVCacheMethodCls: type
    exclude_modules: list[str]
    def __init__(self, exclude_modules: list[str]) -> None: ...
    def is_layer_excluded(self, prefix: str) -> bool: ...
    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None: ...
    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper): ...
    @staticmethod
    def get_config_filenames() -> list[str]: ...
    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ModelOptQuantConfigBase: ...

class ModelOptFp8Config(ModelOptQuantConfigBase):
    quant_method: Incomplete
    is_checkpoint_fp8_serialized: Incomplete
    kv_cache_quant_method: Incomplete
    LinearMethodCls: Incomplete
    def __init__(
        self,
        quant_method: str,
        is_checkpoint_fp8_serialized: bool,
        kv_cache_quant_method: str | None,
        exclude_modules: list[str],
    ) -> None: ...
    def get_name(self) -> QuantizationMethods: ...
    def get_supported_act_dtypes(self) -> list[torch.dtype]: ...
    @classmethod
    def get_min_capability(cls) -> int: ...
    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant
    ) -> QuantizationMethods | None: ...

class ModelOptFp8LinearMethod(LinearMethodBase):
    quant_config: Incomplete
    fp8_linear: Incomplete
    def __init__(self, quant_config: ModelOptFp8Config) -> None: ...
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

class ModelOptFp8PcPtLinearMethod(LinearMethodBase):
    quant_config: Incomplete
    fp8_linear: Incomplete
    def __init__(self, quant_config: ModelOptFp8Config) -> None: ...
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

class ModelOptFp8PbWoLinearMethod(LinearMethodBase):
    quant_config: Incomplete
    weight_block_size: Incomplete
    w8a8_block_fp8_linear: Incomplete
    def __init__(self, quant_config: ModelOptFp8Config) -> None: ...
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

class ModelOptFp8MoEMethod(FusedMoEMethodBase):
    quant_config: Incomplete
    def __init__(
        self, quant_config: ModelOptFp8Config, moe_config: FusedMoEConfig
    ) -> None: ...
    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> mk.FusedMoEPrepareAndFinalizeModular | None: ...
    def select_gemm_impl(
        self,
        prepare_finalize: mk.FusedMoEPrepareAndFinalizeModular,
        layer: torch.nn.Module,
    ) -> mk.FusedMoEExpertsModular: ...
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
    ) -> FusedMoEQuantConfig: ...
    def apply_monolithic(
        self, layer: FusedMoE, x: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...
    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...

class ModelOptNvFp4Config(ModelOptQuantConfigBase):
    is_checkpoint_nvfp4_serialized: Incomplete
    group_size: Incomplete
    kv_cache_quant_algo: Incomplete
    def __init__(
        self,
        is_checkpoint_nvfp4_serialized: bool,
        kv_cache_quant_algo: str | None,
        exclude_modules: list[str],
        group_size: int = 16,
    ) -> None: ...
    def get_name(self) -> QuantizationMethods: ...
    def get_supported_act_dtypes(self) -> list[torch.dtype]: ...
    @classmethod
    def get_min_capability(cls) -> int: ...
    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant
    ) -> QuantizationMethods | None: ...

class ModelOptNvFp4LinearMethod(LinearMethodBase):
    quant_config: Incomplete
    marlin_input_dtype: Incomplete
    backend: Incomplete
    def __init__(self, quant_config: ModelOptNvFp4Config) -> None: ...
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

class ModelOptNvFp4FusedMoE(FusedMoEMethodBase):
    quant_config: Incomplete
    use_global_sf: Incomplete
    def __init__(
        self, quant_config: ModelOptNvFp4Config, moe_config: FusedMoEConfig
    ) -> None: ...
    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> mk.FusedMoEPrepareAndFinalizeModular | None: ...
    def uses_weight_scale_2_pattern(self) -> bool: ...
    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ): ...
    moe_quant_config: Incomplete
    moe_kernel: Incomplete
    def process_weights_after_loading(self, layer: FusedMoE) -> None: ...
    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig: ...
    @property
    def supports_eplb(self) -> bool: ...
    def apply_monolithic(
        self, layer: FusedMoE, x: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...
    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...

class ModelOptMxFp8Config(ModelOptQuantConfigBase):
    is_checkpoint_mxfp8_serialized: Incomplete
    kv_cache_quant_algo: Incomplete
    def __init__(
        self,
        is_checkpoint_mxfp8_serialized: bool,
        kv_cache_quant_algo: str | None,
        exclude_modules: list[str],
    ) -> None: ...
    def get_name(self) -> QuantizationMethods: ...
    def get_supported_act_dtypes(self) -> list[torch.dtype]: ...
    @classmethod
    def get_min_capability(cls) -> int: ...
    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant
    ) -> QuantizationMethods | None: ...

class ModelOptMxFp8LinearMethod(LinearMethodBase):
    quant_config: Incomplete
    backend: Mxfp8LinearBackend
    mxfp8_linear_op: Incomplete
    def __init__(self, quant_config: ModelOptMxFp8Config) -> None: ...
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

class ModelOptMxFp8FusedMoE(FusedMoEMethodBase):
    quant_config: Incomplete
    mxfp8_backend: Incomplete
    def __init__(
        self, quant_config: ModelOptMxFp8Config, moe_config: FusedMoEConfig
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
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None: ...
    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> mk.FusedMoEPrepareAndFinalizeModular | None: ...
    def select_gemm_impl(
        self,
        prepare_finalize: mk.FusedMoEPrepareAndFinalizeModular,
        layer: torch.nn.Module,
    ) -> mk.FusedMoEExpertsModular: ...
    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None: ...
    @property
    def is_monolithic(self) -> bool: ...
    def apply_monolithic(
        self, layer: FusedMoE, x: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...
    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...

class ModelOptMixedPrecisionConfig(ModelOptQuantConfigBase):
    kv_cache_quant_method: Incomplete
    quantized_layers: Incomplete
    fp8_config: Incomplete
    nvfp4_config: Incomplete
    def __init__(
        self,
        kv_cache_quant_method: str | None,
        exclude_modules: list[str],
        quantized_layers: dict[str, dict[str, Any]],
        fp8_config: ModelOptFp8Config,
        nvfp4_config: ModelOptNvFp4Config,
    ) -> None: ...
    def get_name(self) -> QuantizationMethods: ...
    def get_supported_act_dtypes(self) -> list[torch.dtype]: ...
    @classmethod
    def get_min_capability(cls) -> int: ...
    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant
    ) -> QuantizationMethods | None: ...
    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None: ...
    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper): ...
