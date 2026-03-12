import torch
from _typeshed import Incomplete
from collections.abc import Callable as Callable
from vllm.distributed.eplb.eplb_state import EplbLayerState as EplbLayerState
from vllm.model_executor.layers.fused_moe.config import (
    RoutingMethodType as RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.router.base_router import (
    BaseRouter as BaseRouter,
)

class CustomRoutingRouter(BaseRouter):
    custom_routing_function: Incomplete
    renormalize: Incomplete
    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        eplb_state: EplbLayerState,
        custom_routing_function: Callable,
        renormalize: bool = True,
        enable_eplb: bool = False,
        indices_type_getter: Callable[[], torch.dtype | None] | None = None,
    ) -> None: ...
    @property
    def routing_method_type(self) -> RoutingMethodType: ...
