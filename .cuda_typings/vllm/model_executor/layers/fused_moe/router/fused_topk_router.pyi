import torch
from _typeshed import Incomplete
from collections.abc import Callable as Callable
from vllm._aiter_ops import rocm_aiter_ops as rocm_aiter_ops
from vllm.distributed.eplb.eplb_state import EplbLayerState as EplbLayerState
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
) -> tuple[torch.Tensor, ...]: ...
def vllm_topk_sigmoid(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> tuple[torch.Tensor, ...]: ...
def dispatch_topk_softmax_func(
    use_rocm_aiter: bool = False,
) -> Callable[..., tuple[torch.Tensor, ...]]: ...
def dispatch_topk_sigmoid_func(
    use_rocm_aiter: bool = False,
) -> Callable[..., tuple[torch.Tensor, ...]]: ...
def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: torch.dtype | None = None,
    scoring_func: str = "softmax",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

class FusedTopKRouter(BaseRouter):
    renormalize: Incomplete
    scoring_func: Incomplete
    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        eplb_state: EplbLayerState,
        scoring_func: str = "softmax",
        renormalize: bool = True,
        enable_eplb: bool = False,
        indices_type_getter: Callable[[], torch.dtype | None] | None = None,
    ) -> None: ...
    @property
    def routing_method_type(self) -> RoutingMethodType: ...
