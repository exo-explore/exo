import abc
import torch
from abc import ABC, abstractmethod

class MoERunner(ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    def forward(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...
    @abstractmethod
    def must_reduce_shared_expert_outputs(self) -> bool: ...
    @abstractmethod
    def maybe_all_reduce_tensor_model_parallel(
        self, final_hidden_states: torch.Tensor
    ): ...
