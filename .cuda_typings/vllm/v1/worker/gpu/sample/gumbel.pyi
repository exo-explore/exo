import torch
from vllm.triton_utils import tl as tl, triton as triton

def apply_temperature(
    logits: torch.Tensor, expanded_idx_mapping: torch.Tensor, temperature: torch.Tensor
) -> None: ...
def gumbel_sample(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    pos: torch.Tensor,
    apply_temperature: bool,
) -> torch.Tensor: ...
