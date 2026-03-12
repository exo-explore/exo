import torch
from _typeshed import Incomplete
from dataclasses import dataclass
from enum import IntEnum
from triton_kernels.matmul_ogs import PrecisionConfig
from vllm.config import ParallelConfig as ParallelConfig
from vllm.distributed import (
    get_dp_group as get_dp_group,
    get_pcp_group as get_pcp_group,
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation as MoEActivation,
)
from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import (
    OCP_MX_DTYPES as OCP_MX_DTYPES,
    OCP_MX_Scheme as OCP_MX_Scheme,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape as GroupShape,
)
from vllm.platforms import current_platform as current_platform
from vllm.utils.import_utils import has_triton_kernels as has_triton_kernels
from vllm.utils.math_utils import cdiv as cdiv

logger: Incomplete

class RoutingMethodType(IntEnum):
    Default = (0,)
    Renormalize = (1,)
    DeepSeekV3 = (2,)
    Llama4 = (3,)
    RenormalizeNaive = (4,)
    TopK = (5,)
    Custom = (6,)
    Simulated = (7,)
    Unspecified = 8.0

def get_routing_method_type(
    scoring_func: str,
    top_k: int,
    renormalize: bool,
    num_expert_group: int | None,
    has_e_score_bias: bool,
) -> RoutingMethodType: ...
@dataclass
class FusedMoEQuantDesc:
    dtype: torch.dtype | str | None = ...
    shape: GroupShape | None = ...
    scale: torch.Tensor | PrecisionConfig | None = ...
    alpha_or_gscale: torch.Tensor | None = ...
    zp: torch.Tensor | None = ...
    bias: torch.Tensor | None = ...

@dataclass
class FusedMoEQuantConfig:
    is_nvfp4_scale_swizzled: bool = ...
    def __post_init__(self) -> None: ...
    @property
    def quant_dtype(self) -> torch.dtype | str | None: ...
    @property
    def weight_quant_dtype(self) -> torch.dtype | str | None: ...
    @property
    def is_quantized(self) -> bool: ...
    @property
    def is_per_act_token(self) -> bool: ...
    @property
    def per_act_token_quant(self) -> bool: ...
    @property
    def per_out_ch_quant(self) -> bool: ...
    @property
    def is_per_tensor(self) -> bool: ...
    @property
    def block_shape(self) -> list[int] | None: ...
    @property
    def is_block_quantized(self) -> bool: ...
    @property
    def a1_scale(self) -> torch.Tensor | None: ...
    @property
    def a1_gscale(self) -> torch.Tensor | None: ...
    @property
    def a2_scale(self) -> torch.Tensor | None: ...
    @property
    def a2_gscale(self) -> torch.Tensor | None: ...
    @property
    def w1_scale(self) -> torch.Tensor | None: ...
    @property
    def w1_zp(self) -> torch.Tensor | None: ...
    @property
    def w1_bias(self) -> torch.Tensor | None: ...
    @property
    def w1_precision(self) -> PrecisionConfig | None: ...
    @property
    def g1_alphas(self) -> torch.Tensor | None: ...
    @property
    def w2_scale(self) -> torch.Tensor | None: ...
    @property
    def w2_zp(self) -> torch.Tensor | None: ...
    @property
    def w2_bias(self) -> torch.Tensor | None: ...
    @property
    def w2_precision(self) -> PrecisionConfig | None: ...
    @property
    def g2_alphas(self) -> torch.Tensor | None: ...
    @property
    def use_fp8_w8a8(self) -> bool: ...
    @property
    def use_int8_w8a8(self) -> bool: ...
    @property
    def use_int8_w8a16(self) -> bool: ...
    @property
    def use_fp8_w8a16(self) -> bool: ...
    @property
    def use_int4_w4a16(self) -> bool: ...
    @property
    def use_nvfp4_w4a16(self) -> bool: ...
    @property
    def ocp_mx_scheme(self) -> str | None: ...
    @property
    def use_mxfp4_w4a16(self) -> bool: ...
    @property
    def use_mxfp4_w4a4(self) -> bool: ...
    @property
    def use_nvfp4_w4a4(self) -> bool: ...
    @property
    def use_mxfp4_w4a8(self) -> bool: ...
    def config_name(self, dtype: torch.dtype) -> str | None: ...
    def scale_shape(
        self, max_tokens: int, hidden_dim: int
    ) -> tuple[int, int] | None: ...
    def batched_scale_shape(
        self, num_experts: int, max_tokens: int, hidden_dim: int
    ) -> tuple[int, int, int] | None: ...
    @staticmethod
    def make(
        quant_dtype: torch.dtype | str | None = None,
        per_act_token_quant: bool = False,
        per_out_ch_quant: bool = False,
        block_shape: list[int] | None = None,
        w1_scale: torch.Tensor | PrecisionConfig | None = None,
        w2_scale: torch.Tensor | PrecisionConfig | None = None,
        a1_scale: torch.Tensor | None = None,
        a2_scale: torch.Tensor | None = None,
        g1_alphas: torch.Tensor | None = None,
        g2_alphas: torch.Tensor | None = None,
        a1_gscale: torch.Tensor | None = None,
        a2_gscale: torch.Tensor | None = None,
        w1_bias: torch.Tensor | None = None,
        w2_bias: torch.Tensor | None = None,
        w1_zp: torch.Tensor | None = None,
        w2_zp: torch.Tensor | None = None,
        weight_dtype: torch.dtype | str | None = None,
        is_nvfp4_scale_swizzled: bool = True,
    ) -> FusedMoEQuantConfig: ...

def fp8_w8a8_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    per_act_token_quant: bool = False,
    per_out_ch_quant: bool = False,
    block_shape: list[int] | None = None,
    a1_gscale: torch.Tensor | None = None,
    a2_gscale: torch.Tensor | None = None,
    g1_alphas: torch.Tensor | None = None,
    g2_alphas: torch.Tensor | None = None,
) -> FusedMoEQuantConfig: ...
def int8_w8a8_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    a1_scale: torch.Tensor | None,
    a2_scale: torch.Tensor | None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    per_act_token_quant: bool = False,
) -> FusedMoEQuantConfig: ...
def gptq_marlin_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    weight_bits: int,
    group_size: int,
    w1_zp: torch.Tensor | None = None,
    w2_zp: torch.Tensor | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
): ...
def mxfp4_w4a16_moe_quant_config(
    w1_scale: torch.Tensor | PrecisionConfig,
    w2_scale: torch.Tensor | PrecisionConfig,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> FusedMoEQuantConfig: ...
def mxfp4_mxfp8_moe_quant_config(
    w1_scale: torch.Tensor | PrecisionConfig,
    w2_scale: torch.Tensor | PrecisionConfig,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    block_shape: list[int] | None = None,
) -> FusedMoEQuantConfig: ...
def mxfp4_w4a8_moe_quant_config(
    w1_scale: torch.Tensor | PrecisionConfig,
    w2_scale: torch.Tensor | PrecisionConfig,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    block_shape: list[int] | None = None,
) -> FusedMoEQuantConfig: ...
def ocp_mx_moe_quant_config(
    quant_dtype: str,
    w1_scale: torch.Tensor | PrecisionConfig,
    w2_scale: torch.Tensor | PrecisionConfig,
    weight_dtype: str | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    block_shape: list[int] | None = None,
) -> FusedMoEQuantConfig: ...
def nvfp4_moe_quant_config(
    g1_alphas: torch.Tensor,
    g2_alphas: torch.Tensor,
    a1_gscale: torch.Tensor,
    a2_gscale: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    is_nvfp4_scale_swizzled: bool = True,
) -> FusedMoEQuantConfig: ...
def nvfp4_w4a16_moe_quant_config(
    g1_alphas: torch.Tensor,
    g2_alphas: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> FusedMoEQuantConfig: ...
def int4_w4a16_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w1_zp: torch.Tensor | None,
    w2_zp: torch.Tensor | None,
    block_shape: list[int] | None = None,
) -> FusedMoEQuantConfig: ...
def fp8_w8a16_moe_quant_config(
    w1_scale: torch.Tensor, w2_scale: torch.Tensor, block_shape: list[int] | None = None
) -> FusedMoEQuantConfig: ...
def int8_w8a16_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w1_zp: torch.Tensor | None,
    w2_zp: torch.Tensor | None,
    block_shape: list[int] | None = None,
) -> FusedMoEQuantConfig: ...
def int4_w4afp8_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    g1_alphas: torch.Tensor,
    g2_alphas: torch.Tensor,
    per_act_token_quant: bool = False,
    per_out_ch_quant: bool = False,
    block_shape: list[int] | None = None,
) -> FusedMoEQuantConfig: ...
def awq_marlin_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w1_zp: torch.Tensor | None,
    w2_zp: torch.Tensor | None,
    weight_bits: int,
    group_size: int,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> FusedMoEQuantConfig: ...
def biased_moe_quant_config(
    w1_bias: torch.Tensor | None, w2_bias: torch.Tensor | None
) -> FusedMoEQuantConfig: ...

FUSED_MOE_UNQUANTIZED_CONFIG: FusedMoEQuantConfig

@dataclass
class FusedMoEParallelConfig:
    tp_size: int
    pcp_size: int
    dp_size: int
    ep_size: int
    tp_rank: int
    pcp_rank: int
    dp_rank: int
    ep_rank: int
    sp_size: int
    use_ep: bool
    all2all_backend: str
    enable_eplb: bool
    @property
    def is_sequence_parallel(self) -> bool: ...
    @property
    def use_all2all_kernels(self): ...
    @property
    def use_deepep_ht_kernels(self): ...
    @property
    def use_deepep_ll_kernels(self): ...
    @property
    def use_fi_all2allv_kernels(self): ...
    @property
    def use_batched_activation_format(self): ...
    @property
    def use_naive_all2all_kernels(self): ...
    @property
    def use_mori_kernels(self): ...
    @staticmethod
    def flatten_tp_across_dp_and_pcp(
        tp_size: int, dp_size: int, dp_rank: int, pcp_size: int, pcp_rank: int
    ) -> tuple[int, int]: ...
    @staticmethod
    def make(
        tp_size_: int,
        pcp_size_: int,
        dp_size_: int,
        sp_size_: int,
        vllm_parallel_config: ParallelConfig,
    ) -> FusedMoEParallelConfig: ...
    @classmethod
    def make_no_parallel(cls) -> FusedMoEParallelConfig: ...

@dataclass
class FusedMoEConfig:
    num_experts: int
    experts_per_token: int
    hidden_dim: int
    intermediate_size_per_partition: int
    num_local_experts: int
    num_logical_experts: int
    activation: MoEActivation
    device: torch.device | str
    routing_method: RoutingMethodType
    moe_parallel_config: FusedMoEParallelConfig
    in_dtype: torch.dtype
    router_logits_dtype: torch.dtype | None = ...
    moe_backend: str = ...
    max_num_tokens: int = ...
    has_bias: bool = ...
    is_act_and_mul: bool = ...
    is_lora_enabled: bool = ...
    disable_inplace: bool = ...
    def __post_init__(self) -> None: ...
    @property
    def tp_size(self): ...
    @property
    def dp_size(self): ...
    @property
    def pcp_size(self): ...
    @property
    def ep_size(self): ...
    @property
    def sp_size(self): ...
    @property
    def is_sequence_parallel(self): ...
    @property
    def tp_rank(self): ...
    @property
    def dp_rank(self): ...
    @property
    def pcp_rank(self): ...
    @property
    def ep_rank(self): ...
    @property
    def use_ep(self): ...
    @property
    def use_deepep_ht_kernels(self): ...
    @property
    def use_deepep_ll_kernels(self): ...
    @property
    def use_mori_kernels(self): ...
    @property
    def use_fi_all2allv_kernels(self): ...
    @property
    def use_naive_all2all_kernels(self): ...
