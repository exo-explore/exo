import torch
from vllm.platforms import current_platform as current_platform

def batched_count_greater_than(
    x: torch.Tensor, values: torch.Tensor
) -> torch.Tensor: ...
