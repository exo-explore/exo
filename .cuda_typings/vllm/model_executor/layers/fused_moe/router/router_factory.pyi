import torch
from collections.abc import Callable as Callable
from vllm.distributed.eplb.eplb_state import EplbLayerState as EplbLayerState
from vllm.model_executor.layers.fused_moe.config import (
    RoutingMethodType as RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.router.custom_routing_router import (
    CustomRoutingRouter as CustomRoutingRouter,
)
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter as FusedMoERouter,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    FusedTopKBiasRouter as FusedTopKBiasRouter,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    FusedTopKRouter as FusedTopKRouter,
)
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    GroupedTopKRouter as GroupedTopKRouter,
)
from vllm.model_executor.layers.fused_moe.router.routing_simulator_router import (
    RoutingSimulatorRouter as RoutingSimulatorRouter,
)

EMPTY_EPLB_STATE: EplbLayerState

def create_fused_moe_router(
    top_k: int,
    global_num_experts: int,
    renormalize: bool = True,
    indices_type_getter: Callable[[], torch.dtype | None] | None = None,
    use_grouped_topk: bool = False,
    num_expert_group: int | None = None,
    topk_group: int | None = None,
    scoring_func: str = "softmax",
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    custom_routing_function: Callable | None = None,
    enable_eplb: bool = False,
    eplb_state: EplbLayerState = ...,
) -> FusedMoERouter: ...
