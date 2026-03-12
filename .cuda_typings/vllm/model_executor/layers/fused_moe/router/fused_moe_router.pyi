import abc
import torch
from abc import ABC, abstractmethod
from vllm.model_executor.layers.fused_moe.config import (
    RoutingMethodType as RoutingMethodType,
)

class FusedMoERouter(ABC, metaclass=abc.ABCMeta):
    @property
    @abstractmethod
    def routing_method_type(self) -> RoutingMethodType: ...
    @abstractmethod
    def select_experts(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
