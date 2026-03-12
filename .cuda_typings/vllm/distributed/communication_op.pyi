import torch
from .parallel_state import get_tp_group as get_tp_group
from typing import Any

def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor: ...
def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor: ...
def tensor_model_parallel_reduce_scatter(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor: ...
def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> torch.Tensor | None: ...
def broadcast_tensor_dict(
    tensor_dict: dict[Any, torch.Tensor | Any] | None = None, src: int = 0
): ...
