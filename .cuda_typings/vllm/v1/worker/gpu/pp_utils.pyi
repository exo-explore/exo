import torch
from vllm.distributed.parallel_state import get_pp_group as get_pp_group

def pp_broadcast(
    sampled_token_ids: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
) -> None: ...
def pp_receive(
    num_reqs: int, max_sample_len: int = 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
