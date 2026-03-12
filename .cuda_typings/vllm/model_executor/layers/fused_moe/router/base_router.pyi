import abc
import torch
from _typeshed import Incomplete
from collections.abc import Callable as Callable
from vllm.distributed.eplb.eplb_state import EplbLayerState as EplbLayerState
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter as FusedMoERouter,
)
from vllm.platforms import current_platform as current_platform

def eplb_map_to_physical_and_record(
    topk_ids: torch.Tensor,
    expert_load_view: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
    logical_replica_count: torch.Tensor,
) -> torch.Tensor: ...

class BaseRouter(FusedMoERouter, metaclass=abc.ABCMeta):
    top_k: Incomplete
    global_num_experts: Incomplete
    eplb_state: Incomplete
    enable_eplb: Incomplete
    indices_type_getter: Incomplete
    capture_fn: Callable[[torch.Tensor], None] | None
    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        eplb_state: EplbLayerState,
        enable_eplb: bool = False,
        indices_type_getter: Callable[[], torch.dtype | None] | None = None,
    ) -> None: ...
    def set_capture_fn(
        self, capture_fn: Callable[[torch.Tensor], None] | None
    ) -> None: ...
    def select_experts(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
