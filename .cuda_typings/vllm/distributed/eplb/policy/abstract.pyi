import abc
import torch
from abc import ABC, abstractmethod

class AbstractEplbPolicy(ABC, metaclass=abc.ABCMeta):
    @classmethod
    @abstractmethod
    def rebalance_experts(
        cls,
        weight: torch.Tensor,
        num_replicas: int,
        num_groups: int,
        num_nodes: int,
        num_ranks: int,
        old_global_expert_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
