import torch

__all__ = ["reorder_w1w3_to_w3w1"]

def reorder_w1w3_to_w3w1(
    weight: torch.Tensor, scale: torch.Tensor, dim: int = -2
) -> tuple[torch.Tensor, torch.Tensor]: ...
