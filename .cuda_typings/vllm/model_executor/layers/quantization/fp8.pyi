import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from _typeshed import Incomplete
from torch.nn import Module as Module
from torch.utils._python_dispatch import TorchDispatchMode
from typing import Any
from vllm._aiter_ops import rocm_aiter_ops as rocm_aiter_ops
from vllm.distributed import (
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.kernels.linear import (
    init_fp8_linear_kernel as init_fp8_linear_kernel,
)
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant as vllm_is_batch_invariant,
)
from vllm.model_executor.layers.fused_moe import (
    FusedMoE as FusedMoE,
    FusedMoEMethodBase as FusedMoEMethodBase,
    FusedMoeWeightScaleSupported as FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig as FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.layer import (
    UnquantizedFusedMoEMethod as UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    convert_to_fp8_moe_kernel_format as convert_to_fp8_moe_kernel_format,
    make_fp8_moe_kernel as make_fp8_moe_kernel,
    make_fp8_moe_quant_config as make_fp8_moe_quant_config,
    select_fp8_moe_backend as select_fp8_moe_backend,
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
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    W8A8BlockFp8LinearOp as W8A8BlockFp8LinearOp,
    create_fp8_input_scale as create_fp8_input_scale,
    create_fp8_scale_parameter as create_fp8_scale_parameter,
    create_fp8_weight_parameter as create_fp8_weight_parameter,
    maybe_post_process_fp8_weight_block as maybe_post_process_fp8_weight_block,
    process_fp8_input_tensor_strategy_moe as process_fp8_input_tensor_strategy_moe,
    process_fp8_weight_block_strategy as process_fp8_weight_block_strategy,
    process_fp8_weight_tensor_strategy as process_fp8_weight_tensor_strategy,
    process_fp8_weight_tensor_strategy_moe as process_fp8_weight_tensor_strategy_moe,
    validate_fp8_block_shape as validate_fp8_block_shape,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    get_marlin_input_dtype as get_marlin_input_dtype,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear as apply_fp8_marlin_linear,
    prepare_fp8_layer_for_marlin as prepare_fp8_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape as GroupShape,
    is_layer_skipped as is_layer_skipped,
    kFp8Dynamic128Sym as kFp8Dynamic128Sym,
    kFp8DynamicTensorSym as kFp8DynamicTensorSym,
    kFp8DynamicTokenSym as kFp8DynamicTokenSym,
    kFp8Static128BlockSym as kFp8Static128BlockSym,
    kFp8StaticTensorSym as kFp8StaticTensorSym,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    cutlass_block_fp8_supported as cutlass_block_fp8_supported,
    cutlass_fp8_supported as cutlass_fp8_supported,
    normalize_e4m3fn_to_e4m3fnuz as normalize_e4m3fn_to_e4m3fnuz,
)
from vllm.model_executor.model_loader.weight_utils import (
    initialize_single_dummy_weight as initialize_single_dummy_weight,
)
from vllm.model_executor.models.utils import WeightsMapper as WeightsMapper
from vllm.model_executor.parameter import (
    BlockQuantScaleParameter as BlockQuantScaleParameter,
    ModelWeightParameter as ModelWeightParameter,
    PerTensorScaleParameter as PerTensorScaleParameter,
)
from vllm.model_executor.utils import (
    replace_parameter as replace_parameter,
    set_weight_attrs as set_weight_attrs,
)
from vllm.platforms import current_platform as current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported as is_deep_gemm_supported

ACTIVATION_SCHEMES: Incomplete
logger: Incomplete

class Fp8Config(QuantizationConfig):
    is_checkpoint_fp8_serialized: Incomplete
    activation_scheme: Incomplete
    ignored_layers: Incomplete
    weight_block_size: Incomplete
    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: list[str] | None = None,
        weight_block_size: list[int] | None = None,
    ) -> None: ...
    @classmethod
    def get_name(cls) -> QuantizationMethods: ...
    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]: ...
    @classmethod
    def get_min_capability(cls) -> int: ...
    @classmethod
    def get_config_filenames(cls) -> list[str]: ...
    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper): ...
    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Fp8Config: ...
    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None: ...
    def get_cache_scale(self, name: str) -> str | None: ...

class CopyNumelCounter(TorchDispatchMode):
    copied_numel: int
    def __init__(self) -> None: ...
    def __torch_dispatch__(self, func, types, args=(), kwargs=None): ...

class Fp8LinearMethod(LinearMethodBase):
    quant_config: Incomplete
    cutlass_block_fp8_supported: Incomplete
    out_dtype: Incomplete
    marlin_input_dtype: Incomplete
    use_marlin: Incomplete
    use_aiter_and_is_supported: Incomplete
    use_deep_gemm: Incomplete
    weight_block_size: Incomplete
    block_quant: Incomplete
    act_q_static: Incomplete
    w8a8_block_fp8_linear: Incomplete
    fp8_linear: Incomplete
    def __init__(self, quant_config: Fp8Config) -> None: ...
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
    def process_weights_after_loading(self, layer: Module) -> None: ...
    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor: ...

class Fp8OnlineLinearMethod(Fp8LinearMethod):
    uses_meta_device: bool
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
    def process_weights_after_loading(self, layer: Module) -> None: ...

class Fp8MoEMethod(FusedMoEMethodBase):
    quant_config: Incomplete
    weight_block_size: Incomplete
    block_quant: bool
    weight_scale_name: Incomplete
    def __init__(self, quant_config: Fp8Config, layer: torch.nn.Module) -> None: ...
    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ): ...
    def process_weights_after_loading(self, layer: Module) -> None: ...
    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> mk.FusedMoEPrepareAndFinalizeModular | None: ...
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

class Fp8OnlineMoEMethod(Fp8MoEMethod):
    uses_meta_device: bool
    def __init__(self, quant_config: Fp8Config, layer: torch.nn.Module) -> None: ...
    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ): ...
    def process_weights_after_loading(self, layer: Module) -> None: ...

class Fp8KVCacheMethod(BaseKVCacheMethod):
    def __init__(self, quant_config: Fp8Config) -> None: ...
