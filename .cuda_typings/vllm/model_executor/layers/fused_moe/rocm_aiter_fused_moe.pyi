import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from _typeshed import Incomplete
from enum import IntEnum
from vllm._aiter_ops import rocm_aiter_ops as rocm_aiter_ops
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
    kFp8Dynamic128Sym as kFp8Dynamic128Sym,
    kFp8DynamicTensorSym as kFp8DynamicTensorSym,
    kFp8DynamicTokenSym as kFp8DynamicTokenSym,
    kFp8Static128BlockSym as kFp8Static128BlockSym,
    kFp8StaticChannelSym as kFp8StaticChannelSym,
    kFp8StaticTensorSym as kFp8StaticTensorSym,
)

class QuantMethod(IntEnum):
    NO = 0
    PER_TENSOR = 1
    PER_TOKEN = 2
    BLOCK_1X32 = 3
    BLOCK_1X128 = 4
    BLOCK_128x128 = 5

class ActivationMethod(IntEnum):
    SILU = 0
    GELU = 1

aiter_topK_meta_data: Incomplete

def init_aiter_topK_meta_data(
    n_routed_experts: int,
    n_shared_experts: int,
    top_k: int,
    tp_rank: int,
    tp_size: int,
    shared_experts_score: float = 1.0,
    max_num_tokens: int = 32768,
    is_EP: bool = False,
): ...
def rocm_aiter_grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    num_fused_shared_experts: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def rocm_aiter_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: MoEActivation = ...,
    apply_router_weight_on_input: bool = False,
    expert_map: torch.Tensor | None = None,
    quant_config: FusedMoEQuantConfig | None = None,
    a1q_scale: torch.Tensor | None = None,
    num_local_tokens: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor: ...

class AiterExperts(mk.FusedMoEExpertsModular):
    @property
    def expects_unquantized_inputs(self) -> bool: ...
    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat: ...
    def supports_expert_map(self): ...
    def supports_chunking(self): ...
    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce: ...
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
