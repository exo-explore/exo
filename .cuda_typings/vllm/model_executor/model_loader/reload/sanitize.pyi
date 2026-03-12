import torch

__all__ = ["sanitize_layer_refs", "restore_layer_refs"]

def sanitize_layer_refs(
    tensor: torch.Tensor, layer: torch.nn.Module
) -> torch.Tensor: ...
def restore_layer_refs(
    tensor: torch.Tensor, layer: torch.nn.Module
) -> torch.Tensor: ...
