import torch
from _typeshed import Incomplete
from collections.abc import Callable as Callable
from vllm._aiter_ops import rocm_aiter_ops as rocm_aiter_ops
from vllm.distributed.eplb.eplb_state import EplbLayerState as EplbLayerState
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant as vllm_is_batch_invariant,
)
from vllm.model_executor.layers.fused_moe.config import (
    RoutingMethodType as RoutingMethodType,
    get_routing_method_type as get_routing_method_type,
)
from vllm.model_executor.layers.fused_moe.router.base_router import (
    BaseRouter as BaseRouter,
)

def vllm_topk_softmax(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    e_score_correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]: ...
def vllm_topk_sigmoid(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    e_score_correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]: ...
def fused_topk_bias(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    scoring_func: str = "softmax",
    indices_type: torch.dtype | None = None,
): ...

class FusedTopKBiasRouter(BaseRouter):
    e_score_correction_bias: Incomplete
    renormalize: Incomplete
    scoring_func: Incomplete
    routed_scaling_factor: Incomplete
    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        eplb_state: EplbLayerState,
        e_score_correction_bias: torch.Tensor,
        scoring_func: str,
        renormalize: bool = True,
        routed_scaling_factor: float = 1.0,
        enable_eplb: bool = False,
        indices_type_getter: Callable[[], torch.dtype | None] | None = None,
    ) -> None: ...
    @property
    def routing_method_type(self) -> RoutingMethodType: ...
