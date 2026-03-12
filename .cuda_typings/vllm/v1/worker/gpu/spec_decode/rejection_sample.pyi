import torch
from vllm.triton_utils import tl as tl, triton as triton

def rejection_sample(
    target_sampled: torch.Tensor,
    input_ids: torch.Tensor,
    cu_num_logits: torch.Tensor,
    num_speculative_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]: ...
