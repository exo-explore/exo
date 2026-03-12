import abc
import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from _typeshed import Incomplete
from collections.abc import Callable as Callable
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation as MoEActivation,
    apply_moe_activation as apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig as FusedMoEConfig,
    FusedMoEParallelConfig as FusedMoEParallelConfig,
    FusedMoEQuantConfig as FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    batched_moe_align_block_size as batched_moe_align_block_size,
    moe_align_block_size as moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate as TopKWeightAndReduceDelegate,
    TopKWeightAndReduceNoOP as TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import (
    disable_inplace as disable_inplace,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    get_marlin_input_dtype as get_marlin_input_dtype,
    marlin_make_workspace_new as marlin_make_workspace_new,
    marlin_moe_intermediate_size as marlin_moe_intermediate_size,
    marlin_quant_input as marlin_quant_input,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey as QuantKey,
    kFp8Static128BlockSym as kFp8Static128BlockSym,
    kFp8StaticChannelSym as kFp8StaticChannelSym,
    kFp8StaticTensorSym as kFp8StaticTensorSym,
    kNvfp4Static as kNvfp4Static,
)
from vllm.platforms import current_platform as current_platform
from vllm.scalar_type import ScalarType as ScalarType, scalar_types as scalar_types

def fused_marlin_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    bias1: torch.Tensor | None,
    bias2: torch.Tensor | None,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_type_id: int,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    activation: MoEActivation = ...,
    activation_func: Callable[[MoEActivation, torch.Tensor, torch.Tensor], None] = ...,
    moe_sum: Callable[[torch.Tensor, torch.Tensor], None] | None = None,
    expert_map: torch.Tensor | None = None,
    input_global_scale1: torch.Tensor | None = None,
    input_global_scale2: torch.Tensor | None = None,
    global_scale1: torch.Tensor | None = None,
    global_scale2: torch.Tensor | None = None,
    g_idx1: torch.Tensor | None = None,
    g_idx2: torch.Tensor | None = None,
    sort_indices1: torch.Tensor | None = None,
    sort_indices2: torch.Tensor | None = None,
    w1_zeros: torch.Tensor | None = None,
    w2_zeros: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
    intermediate_cache13: torch.Tensor | None = None,
    intermediate_cache2: torch.Tensor | None = None,
    is_k_full: bool = True,
    output: torch.Tensor | None = None,
    input_dtype: torch.dtype | None = None,
    inplace: bool = False,
) -> torch.Tensor: ...
def batched_fused_marlin_moe(
    hidden_states: torch.Tensor,
    expert_num_tokens: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    bias1: torch.Tensor | None,
    bias2: torch.Tensor | None,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    quant_type_id: int,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    activation: MoEActivation = ...,
    expert_map: torch.Tensor | None = None,
    global_scale1: torch.Tensor | None = None,
    global_scale2: torch.Tensor | None = None,
    g_idx1: torch.Tensor | None = None,
    g_idx2: torch.Tensor | None = None,
    sort_indices1: torch.Tensor | None = None,
    sort_indices2: torch.Tensor | None = None,
    w1_zeros: torch.Tensor | None = None,
    w2_zeros: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
    intermediate_cache13: torch.Tensor | None = None,
    intermediate_cache2: torch.Tensor | None = None,
    is_k_full: bool = True,
    output: torch.Tensor | None = None,
    inplace: bool = False,
) -> torch.Tensor: ...

class MarlinExpertsBase(mk.FusedMoEExpertsModular, metaclass=abc.ABCMeta):
    w13_g_idx: Incomplete
    w2_g_idx: Incomplete
    w13_g_idx_sort_indices: Incomplete
    w2_g_idx_sort_indices: Incomplete
    is_k_full: Incomplete
    input_dtype: Incomplete
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
        w13_g_idx: torch.Tensor | None = None,
        w2_g_idx: torch.Tensor | None = None,
        w13_g_idx_sort_indices: torch.Tensor | None = None,
        w2_g_idx_sort_indices: torch.Tensor | None = None,
        is_k_full: bool = True,
    ) -> None: ...
    @property
    def quant_type_id(self) -> int: ...
    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]: ...

class MarlinExperts(MarlinExpertsBase):
    def supports_expert_map(self) -> bool: ...
    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce: ...
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
    def moe_sum(self, input: torch.Tensor, output: torch.Tensor) -> None: ...

class BatchedMarlinExperts(MarlinExpertsBase):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int,
        num_dispatchers: int,
        w13_g_idx: torch.Tensor | None = None,
        w2_g_idx: torch.Tensor | None = None,
        w13_g_idx_sort_indices: torch.Tensor | None = None,
        w2_g_idx_sort_indices: torch.Tensor | None = None,
        is_k_full: bool = True,
    ) -> None: ...
    def supports_expert_map(self) -> bool: ...
    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce: ...
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
