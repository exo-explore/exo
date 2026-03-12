import abc
import torch
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Callable as Callable
from typing import Any
from vllm.distributed.eplb.eplb_state import EplbLayerState as EplbLayerState
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.fused_moe.config import (
    RoutingMethodType as RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.router.base_router import (
    BaseRouter as BaseRouter,
)

logger: Incomplete

class RoutingStrategy(ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    def route_tokens(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        indices_type: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class DistributionBasedRouting(RoutingStrategy):
    distribution: Incomplete
    distribution_params: Incomplete
    def __init__(
        self, distribution: str = "uniform", **distribution_params: Any
    ) -> None: ...
    def route_tokens(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        indices_type: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def get_distribution_info(self) -> dict: ...

class RoutingSimulator:
    @classmethod
    def register_strategy(cls, name: str, strategy: RoutingStrategy): ...
    @classmethod
    def get_available_strategies(cls) -> list[str]: ...
    @staticmethod
    def simulate_routing(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        strategy_name: str,
        top_k: int,
        indices_type: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class RoutingSimulatorRouter(BaseRouter):
    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        eplb_state: EplbLayerState,
        enable_eplb: bool = False,
        indices_type_getter: Callable[[], torch.dtype | None] | None = None,
    ) -> None: ...
    @property
    def routing_method_type(self) -> RoutingMethodType: ...
