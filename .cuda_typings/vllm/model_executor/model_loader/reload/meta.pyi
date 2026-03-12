import inspect
import torch
from .types import LayerReloadingInfo, LayerTensors
from collections.abc import Callable
from torch.utils._python_dispatch import TorchDispatchMode

__all__ = [
    "to_meta_tensor",
    "materialize_meta_tensor",
    "capture_layer_to_meta",
    "restore_layer_on_meta",
    "materialize_layer",
    "get_numel_loaded",
]

def to_meta_tensor(tensor: torch.Tensor) -> torch.Tensor: ...
def materialize_meta_tensor(meta_tensor: torch.Tensor) -> torch.Tensor: ...
def capture_layer_to_meta(layer: torch.nn.Module) -> LayerTensors: ...
def restore_layer_on_meta(layer: torch.nn.Module, info: LayerReloadingInfo): ...
def materialize_layer(layer: torch.nn.Module) -> None: ...

class MetaCopyCounter(TorchDispatchMode):
    copied_numel: int
    def __init__(self) -> None: ...
    def __torch_dispatch__(self, func, types, args=(), kwargs=None): ...

def get_numel_loaded(
    weight_loader: Callable, args: inspect.BoundArguments
) -> tuple[int, object]: ...
