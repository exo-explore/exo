import abc
import torch
from abc import ABC, abstractmethod

__all__ = ["CompressedTensorsScheme"]

class CompressedTensorsScheme(ABC, metaclass=abc.ABCMeta):
    @classmethod
    @abstractmethod
    def get_min_capability(cls) -> int: ...
    @abstractmethod
    def create_weights(self, *args, **kwargs): ...
    @abstractmethod
    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None
    ): ...
    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module): ...
