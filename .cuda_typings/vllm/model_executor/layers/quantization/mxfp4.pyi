import torch
from _typeshed import Incomplete
from enum import Enum
from vllm import envs as envs
from vllm._aiter_ops import rocm_aiter_ops as rocm_aiter_ops
from vllm.config import get_current_vllm_config as get_current_vllm_config
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.model_executor.layers.fused_moe import (
    FusedMoE as FusedMoE,
    FusedMoEConfig as FusedMoEConfig,
    FusedMoEMethodBase as FusedMoEMethodBase,
    MoEActivation as MoEActivation,
    modular_kernel as mk,
)
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize as maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig as FusedMoEQuantConfig,
    mxfp4_mxfp8_moe_quant_config as mxfp4_mxfp8_moe_quant_config,
    mxfp4_w4a16_moe_quant_config as mxfp4_w4a16_moe_quant_config,
    ocp_mx_moe_quant_config as ocp_mx_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
    BatchedMarlinExperts as BatchedMarlinExperts,
    MarlinExperts as MarlinExperts,
)
from vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe import (
    OAITritonExperts as OAITritonExperts,
    UnfusedOAITritonExperts as UnfusedOAITritonExperts,
)
from vllm.model_executor.layers.fused_moe.trtllm_moe import (
    TrtLlmGenExperts as TrtLlmGenExperts,
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
    get_marlin_input_dtype as get_marlin_input_dtype,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    prepare_moe_fp4_layer_for_marlin as prepare_moe_fp4_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    CK_MXFP4_MOE_DIM_ALIGNMENT as CK_MXFP4_MOE_DIM_ALIGNMENT,
    get_padding_alignment as get_padding_alignment,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped as is_layer_skipped,
)
from vllm.model_executor.utils import set_weight_attrs as set_weight_attrs
from vllm.platforms import current_platform as current_platform
from vllm.utils.flashinfer import has_flashinfer as has_flashinfer
from vllm.utils.import_utils import has_triton_kernels as has_triton_kernels
from vllm.utils.math_utils import round_up as round_up

logger: Incomplete

class Mxfp4Backend(Enum):
    NONE = 0
    SM100_FI_MXFP4_MXFP8_TRTLLM = 1
    SM100_FI_MXFP4_MXFP8_CUTLASS = 2
    SM100_FI_MXFP4_BF16 = 3
    SM90_FI_MXFP4_BF16 = 4
    MARLIN = 5
    TRITON = 6
    CK = 7

def get_mxfp4_backend_with_lora() -> Mxfp4Backend: ...
def get_mxfp4_backend(with_lora_support: bool) -> Mxfp4Backend: ...

class Mxfp4Config(QuantizationConfig):
    ignored_layers: Incomplete
    def __init__(self, ignored_layers: list[str] | None = None) -> None: ...
    @classmethod
    def from_config(cls, config): ...
    @classmethod
    def get_min_capability(cls) -> int: ...
    @classmethod
    def get_name(cls) -> QuantizationMethods: ...
    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]: ...
    @classmethod
    def get_config_filenames(cls) -> list[str]: ...
    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None: ...
    def is_mxfp4_quant(self, prefix: str, layer: torch.nn.Module) -> bool: ...

class Mxfp4MoEMethod(FusedMoEMethodBase):
    weight_dtype: str
    mxfp4_backend: Incomplete
    max_capture_size: Incomplete
    moe_kernel: mk.FusedMoEKernel | None
    def __init__(self, moe: FusedMoEConfig) -> None: ...
    num_experts: Incomplete
    intermediate_size: Incomplete
    hidden_size: Incomplete
    hidden_pad: Incomplete
    intermediate_pad: Incomplete
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
    w13_precision_config: Incomplete
    w2_precision_config: Incomplete
    w13_weight: Incomplete
    w2_weight: Incomplete
    def process_weights_after_loading(self, layer): ...
    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None: ...
    def select_gemm_impl(
        self,
        prepare_finalize: mk.FusedMoEPrepareAndFinalizeModular,
        layer: torch.nn.Module,
    ) -> mk.FusedMoEExpertsModular: ...
    @property
    def is_monolithic(self) -> bool: ...
    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...
    def apply_monolithic(
        self, layer: FusedMoE, x: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...

class XpuMxfp4MoEMethod(Mxfp4MoEMethod):
    moe_config: Incomplete
    def __init__(self, moe_config: FusedMoEConfig) -> None: ...
    original_hidden_size: Incomplete
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
    @property
    def is_monolithic(self) -> bool: ...
    def apply_monolithic(
        self, layer: FusedMoE, x: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor: ...
