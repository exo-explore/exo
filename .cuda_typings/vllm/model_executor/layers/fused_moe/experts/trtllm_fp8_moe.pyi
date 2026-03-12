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
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    activation_to_flashinfer_int as activation_to_flashinfer_int,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey as QuantKey,
    kFp8Dynamic128Sym as kFp8Dynamic128Sym,
    kFp8Static128BlockSym as kFp8Static128BlockSym,
    kFp8StaticTensorSym as kFp8StaticTensorSym,
)
from vllm.platforms import current_platform as current_platform

class TrtLlmFp8Experts(mk.FusedMoEExpertsMonolithic):
    routing_method_type: Incomplete
    topk: Incomplete
    intermediate_size_per_partition: Incomplete
    hidden_dim: Incomplete
    local_num_experts: Incomplete
    ep_rank: Incomplete
    def __init__(
        self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig
    ) -> None: ...
    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat: ...
    def supports_chunking(self) -> bool: ...
    def supports_expert_map(self) -> bool: ...
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
