import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from _typeshed import Incomplete
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation as MoEActivation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig as FusedMoEConfig,
    FusedMoEParallelConfig as FusedMoEParallelConfig,
    FusedMoEQuantConfig as FusedMoEQuantConfig,
    RoutingMethodType as RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP as TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    activation_to_flashinfer_int as activation_to_flashinfer_int,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey as QuantKey,
    kNvfp4Dynamic as kNvfp4Dynamic,
    kNvfp4Static as kNvfp4Static,
)
from vllm.platforms import current_platform as current_platform

class TrtLlmNvFp4ExpertsBase:
    moe_config: Incomplete
    quant_config: Incomplete
    routing_method_type: Incomplete
    topk: Incomplete
    intermediate_size_per_partition: Incomplete
    hidden_dim: Incomplete
    local_num_experts: Incomplete
    ep_rank: Incomplete
    g1_scale_c: Incomplete
    def __init__(
        self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig
    ) -> None: ...
    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat: ...
    def supports_chunking(self) -> bool: ...
    def supports_expert_map(self) -> bool: ...

class TrtLlmNvFp4ExpertsModular(TrtLlmNvFp4ExpertsBase, mk.FusedMoEExpertsModular):
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

class TrtLlmNvFp4ExpertsMonolithic(
    TrtLlmNvFp4ExpertsBase, mk.FusedMoEExpertsMonolithic
):
    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor: ...
