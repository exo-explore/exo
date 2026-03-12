import numpy as np
import torch
from .abstract import AbstractEplbPolicy as AbstractEplbPolicy

class DefaultEplbPolicy(AbstractEplbPolicy):
    @classmethod
    def balanced_packing(
        cls, weight: np.ndarray, num_packs: int
    ) -> tuple[np.ndarray, np.ndarray]: ...
    @classmethod
    def replicate_experts(
        cls, weight: np.ndarray, num_phy: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    @classmethod
    def rebalance_experts_hierarchical(
        cls,
        weight: np.ndarray,
        num_physical_experts: int,
        num_groups: int,
        num_nodes: int,
        num_gpus: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    @classmethod
    def preserve_intragpu_slots(
        cls,
        phy2log: np.ndarray,
        phy_replicas_idx: np.ndarray,
        num_ranks: int,
        old_phy2log: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]: ...
    @classmethod
    def rebalance_experts(
        cls,
        weight: torch.Tensor,
        num_replicas: int,
        num_groups: int,
        num_nodes: int,
        num_ranks: int,
        old_global_expert_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
