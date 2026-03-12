import abc
import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from _typeshed import Incomplete
from triton_kernels.matmul_ogs import GatherIndx, RoutingData, ScatterIndx
from triton_kernels.tensor import Bitmatrix
from vllm._aiter_ops import rocm_aiter_ops as rocm_aiter_ops
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation as MoEActivation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG as FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEParallelConfig as FusedMoEParallelConfig,
    FusedMoEQuantConfig as FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP as TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey as QuantKey,
)
from vllm.platforms import current_platform as current_platform
from vllm.triton_utils import tl as tl, triton as triton
from vllm.utils.import_utils import has_triton_kernels as has_triton_kernels

logger: Incomplete
use_legacy_triton_kernels: bool

@triton.jit
def pack_bitmatrix(
    bitmatrix,
    topk_ids,
    n_rows,
    bm_cols: tl.constexpr,
    n_expts_act,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
): ...
def legacy_routing_from_bitmatrix(
    bitmatrix: Bitmatrix,
    expt_scal: torch.Tensor,
    expt_indx: torch.Tensor,
    n_expts_tot: int,
    n_expts_act: int,
) -> tuple["RoutingData", "GatherIndx", "ScatterIndx"]: ...
def legacy_routing(
    logits: torch.Tensor, n_expts_act: int, sm_first: bool = False
) -> tuple["RoutingData", "GatherIndx", "ScatterIndx"]: ...
def triton_kernel_moe_forward(
    hidden_states: torch.Tensor,
    w1,
    w2,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    activation: MoEActivation = ...,
    quant_config: FusedMoEQuantConfig | None = None,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    unpadded_N_w1=None,
    unpadded_K_w1=None,
    unpadded_N_w2=None,
    unpadded_K_w2=None,
) -> torch.Tensor: ...
def triton_kernel_fused_experts(
    output_tensor: torch.Tensor,
    hidden_states: torch.Tensor,
    w1,
    w2,
    routing_data,
    gather_indx,
    scatter_indx,
    topk: int,
    activation: MoEActivation = ...,
    quant_config: FusedMoEQuantConfig | None = None,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    intermediate_cache: torch.Tensor | None = None,
    a1q_scale: torch.Tensor | None = None,
) -> torch.Tensor: ...
def triton_kernel_fused_mxfp4_w4a8_experts(
    output_tensor: torch.Tensor,
    hidden_states: torch.Tensor,
    w1,
    w2,
    routing_data,
    gather_indx,
    scatter_indx,
    activation: str = "silu",
    quant_config: FusedMoEQuantConfig | None = None,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    a1q_scale: torch.Tensor | None = None,
    unpadded_N_w1=None,
    unpadded_K_w1=None,
    unpadded_N_w2=None,
    unpadded_K_w2=None,
) -> torch.Tensor: ...
def make_routing_data(
    topk_ids: torch.Tensor, topk_weights: torch.Tensor, num_local_experts: int
) -> tuple["RoutingData", torch.Tensor, torch.Tensor]: ...

class BaseOAITritonExperts(mk.FusedMoEExpertsModular, metaclass=abc.ABCMeta):
    def supports_expert_map(self) -> bool: ...
    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]: ...
    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce: ...

class OAITritonExperts(BaseOAITritonExperts):
    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat: ...
    def supports_chunking(self) -> bool: ...
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
    quant_config: FusedMoEQuantConfig
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

class UnfusedOAITritonExperts(BaseOAITritonExperts):
    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat: ...
    def supports_chunking(self) -> bool: ...
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
    def moe_sum(self, input: torch.Tensor, output: torch.Tensor): ...
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
