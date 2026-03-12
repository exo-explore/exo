import torch
from .tensor import Bitmatrix as Bitmatrix

def compaction(yv, yi, bitmask, sentinel: int = -1): ...
def compaction_torch(
    yv: torch.Tensor, yi: torch.Tensor, bitmask: torch.Tensor, sentinel: int = -1
): ...
