import abc
import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from _typeshed import Incomplete
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation as MoEActivation,
    apply_moe_activation as apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig as FusedMoEConfig,
    FusedMoEParallelConfig as FusedMoEParallelConfig,
    FusedMoEQuantConfig as FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    moe_permute as moe_permute,
    moe_unpermute as moe_unpermute,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoDPEPModular as MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate as TopKWeightAndReduceDelegate,
    TopKWeightAndReduceNoOP as TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey as QuantKey,
    kFp8DynamicTensorSym as kFp8DynamicTensorSym,
    kFp8DynamicTokenSym as kFp8DynamicTokenSym,
    kFp8StaticChannelSym as kFp8StaticChannelSym,
    kFp8StaticTensorSym as kFp8StaticTensorSym,
    kNvfp4Dynamic as kNvfp4Dynamic,
    kNvfp4Static as kNvfp4Static,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    cutlass_group_gemm_supported as cutlass_group_gemm_supported,
)
from vllm.platforms import current_platform as current_platform
from vllm.scalar_type import scalar_types as scalar_types

logger: Incomplete

def run_cutlass_moe_fp8(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: MoEActivation,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    a1q_scale: torch.Tensor | None,
    a2_scale: torch.Tensor | None,
    ab_strides1: torch.Tensor,
    ab_strides2: torch.Tensor,
    c_strides1: torch.Tensor,
    c_strides2: torch.Tensor,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
    expert_num_tokens: torch.Tensor | None,
    out_dtype: torch.dtype,
    per_act_token: bool,
    per_out_ch: bool,
    use_batched_format: bool,
    topk_weights: torch.Tensor | None,
): ...

class CutlassExpertsFp8Base(mk.FusedMoEExpertsModular, metaclass=abc.ABCMeta):
    out_dtype: Incomplete
    ab_strides1: Incomplete
    ab_strides2: Incomplete
    c_strides1: Incomplete
    c_strides2: Incomplete
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ) -> None: ...
    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce: ...
    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ): ...

class CutlassExpertsFp8(CutlassExpertsFp8Base):
    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat: ...
    def supports_chunking(self) -> bool: ...
    def supports_expert_map(self) -> bool: ...
    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce: ...
    def workspace_dtype(self, act_dtype: torch.dtype) -> torch.dtype: ...
    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]: ...

class CutlassBatchedExpertsFp8(CutlassExpertsFp8Base):
    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat: ...
    def supports_chunking(self) -> bool: ...
    def supports_expert_map(self) -> bool: ...
    def workspace_dtype(self, act_dtype: torch.dtype) -> torch.dtype: ...
    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]: ...

FLOAT4_E2M1_MAX: Incomplete
FLOAT8_E4M3_MAX: Incomplete

def run_cutlass_moe_fp4(
    output: torch.Tensor,
    a: torch.Tensor,
    a1_gscale: torch.Tensor,
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_alphas: torch.Tensor,
    a2_gscale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_alphas: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: MoEActivation,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
    m: int,
    n: int,
    k: int,
    e: int,
    device: torch.device,
    apply_router_weight_on_input: bool = False,
) -> None: ...

class CutlassExpertsFp4(mk.FusedMoEExpertsModular):
    @property
    def expects_unquantized_inputs(self) -> bool: ...
    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat: ...
    def supports_expert_map(self) -> bool: ...
    def supports_chunking(self) -> bool: ...
    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce: ...
    def workspace_dtype(self, act_dtype: torch.dtype) -> torch.dtype: ...
    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]: ...
    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ): ...

def run_cutlass_moe_w4a8_fp8(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: MoEActivation,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    a1q_scale: torch.Tensor | None,
    a2_scale: torch.Tensor | None,
    w1_chan_scale: torch.Tensor,
    w2_chan_scale: torch.Tensor,
    a_strides1: torch.Tensor,
    a_strides2: torch.Tensor,
    b_strides1: torch.Tensor,
    b_strides2: torch.Tensor,
    c_strides1: torch.Tensor,
    c_strides2: torch.Tensor,
    s_strides1: torch.Tensor,
    s_strides2: torch.Tensor,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
    expert_num_tokens: torch.Tensor | None,
    out_dtype: torch.dtype,
    per_act_token: bool,
    per_out_ch: bool,
    use_batched_format: bool,
    topk_weights: torch.Tensor | None,
    group_size: int,
): ...

class CutlassExpertsW4A8Fp8(mk.FusedMoEExpertsModular):
    out_dtype: Incomplete
    a_strides1: Incomplete
    a_strides2: Incomplete
    b_strides1: Incomplete
    b_strides2: Incomplete
    c_strides1: Incomplete
    c_strides2: Incomplete
    s_strides1: Incomplete
    s_strides2: Incomplete
    group_size: Incomplete
    def __init__(
        self,
        out_dtype: torch.dtype | None,
        a_strides1: torch.Tensor,
        a_strides2: torch.Tensor,
        b_strides1: torch.Tensor,
        b_strides2: torch.Tensor,
        c_strides1: torch.Tensor,
        c_strides2: torch.Tensor,
        s_strides1: torch.Tensor,
        s_strides2: torch.Tensor,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        group_size: int,
    ) -> None: ...
    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat: ...
    def supports_chunking(self) -> bool: ...
    def supports_expert_map(self) -> bool: ...
    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce: ...
    def workspace_dtype(self, act_dtype: torch.dtype) -> torch.dtype: ...
    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]: ...
    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ): ...

def cutlass_moe_w4a8_fp8(
    a: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    a_strides1: torch.Tensor,
    a_strides2: torch.Tensor,
    b_strides1: torch.Tensor,
    b_strides2: torch.Tensor,
    c_strides1: torch.Tensor,
    c_strides2: torch.Tensor,
    s_strides1: torch.Tensor,
    s_strides2: torch.Tensor,
    quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    activation: MoEActivation = ...,
    expert_map: torch.Tensor | None = None,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    group_size: int = 128,
) -> torch.Tensor: ...
