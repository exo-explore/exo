import torch
from vllm.triton_utils import triton as triton
from vllm.utils.math_utils import round_up as round_up

def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
def batched_moe_align_block_size(
    max_tokens_per_batch: int, block_size: int, expert_num_tokens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
