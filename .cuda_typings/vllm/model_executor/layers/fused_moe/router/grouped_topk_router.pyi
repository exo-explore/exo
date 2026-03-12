import torch
from _typeshed import Incomplete
from collections.abc import Callable as Callable
from vllm._aiter_ops import rocm_aiter_ops as rocm_aiter_ops
from vllm.distributed.eplb.eplb_state import EplbLayerState as EplbLayerState
from vllm.model_executor.custom_op import CustomOp as CustomOp
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant as vllm_is_batch_invariant,
)
from vllm.model_executor.layers.fused_moe.config import (
    RoutingMethodType as RoutingMethodType,
    get_routing_method_type as get_routing_method_type,
)
from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
    rocm_aiter_grouped_topk as rocm_aiter_grouped_topk,
)
from vllm.model_executor.layers.fused_moe.router.base_router import (
    BaseRouter as BaseRouter,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    fused_topk_bias as fused_topk_bias,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    fused_topk as fused_topk,
)
from vllm.model_executor.utils import (
    maybe_disable_graph_partition as maybe_disable_graph_partition,
)
from vllm.platforms import current_platform as current_platform

def fused_grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    e_score_correction_bias: torch.Tensor,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]: ...

class GroupedTopk(CustomOp):
    native_impl: Incomplete
    topk: Incomplete
    renormalize: Incomplete
    num_expert_group: Incomplete
    topk_group: Incomplete
    scoring_func: Incomplete
    routed_scaling_factor: Incomplete
    num_fused_shared_experts: Incomplete
    def __init__(
        self,
        topk: int,
        renormalize: bool,
        num_expert_group: int = 0,
        topk_group: int = 0,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        num_fused_shared_experts: int = 0,
    ) -> None: ...
    def forward_native(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        e_score_correction_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        e_score_correction_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        e_score_correction_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class GroupedTopKRouter(BaseRouter):
    num_expert_group: Incomplete
    topk_group: Incomplete
    renormalize: Incomplete
    scoring_func: Incomplete
    routed_scaling_factor: Incomplete
    e_score_correction_bias: Incomplete
    num_fused_shared_experts: Incomplete
    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        eplb_state: EplbLayerState,
        num_expert_group: int,
        topk_group: int,
        renormalize: bool = True,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        num_fused_shared_experts: int = 0,
        enable_eplb: bool = False,
        indices_type_getter: Callable[[], torch.dtype | None] | None = None,
    ) -> None: ...
    @property
    def routing_method_type(self) -> RoutingMethodType: ...
