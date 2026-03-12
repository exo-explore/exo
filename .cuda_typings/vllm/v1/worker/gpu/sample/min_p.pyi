import torch
from vllm.triton_utils import tl as tl, triton as triton

def apply_min_p(
    logits: torch.Tensor, expanded_idx_mapping: torch.Tensor, min_p: torch.Tensor
) -> None: ...
