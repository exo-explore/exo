import torch
from .utils import log_replacement as log_replacement
from _typeshed import Incomplete
from typing import Any
from vllm.config import VllmConfig as VllmConfig
from vllm.config.utils import getattr_iter as getattr_iter
from vllm.distributed import get_dp_group as get_dp_group, get_ep_group as get_ep_group
from vllm.forward_context import (
    ForwardContext as ForwardContext,
    get_forward_context as get_forward_context,
)
from vllm.model_executor.custom_op import CustomOp as CustomOp
from vllm.model_executor.layers.fused_moe import FusedMoE as FusedMoE
from vllm.model_executor.models.interfaces import MixtureOfExperts as MixtureOfExperts
from vllm.model_executor.models.utils import maybe_prefix as maybe_prefix
from vllm.platforms import current_platform as current_platform
from vllm.utils.torch_utils import (
    direct_register_custom_op as direct_register_custom_op,
)

class TransformersFusedMoE(FusedMoE):
    def __init__(self, *args, **kwargs) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor: ...

def transformers_moe_forward(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    layer_name: str,
) -> torch.Tensor: ...
def transformers_moe_forward_fake(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    layer_name: str,
) -> torch.Tensor: ...

class MoEMixin(MixtureOfExperts):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def set_eplb_state(
        self,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ): ...
    num_physical_experts: Incomplete
    num_local_physical_experts: Incomplete
    num_redundant_experts: Incomplete
    def update_physical_experts_metadata(
        self, num_physical_experts: int, num_local_physical_experts: int
    ): ...
    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]: ...
    mlp_moe_layers: Incomplete
    moe_layers: Incomplete
    expert_weights: Incomplete
    num_moe_layers: int
    num_expert_groups: Incomplete
    num_logical_experts: Incomplete
    num_routed_experts: Incomplete
    num_shared_experts: Incomplete
    def recursive_replace(self): ...
