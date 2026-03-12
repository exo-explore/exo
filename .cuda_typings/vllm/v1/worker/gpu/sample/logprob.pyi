import torch
from vllm.triton_utils import tl as tl, triton as triton
from vllm.v1.outputs import LogprobsTensors as LogprobsTensors

def compute_token_logprobs(
    logits: torch.Tensor, token_ids: torch.Tensor
) -> torch.Tensor: ...
def compute_topk_logprobs(
    logits: torch.Tensor,
    num_logprobs: int,
    sampled_token_ids: torch.Tensor,
    cu_num_logits: list[int] | None = None,
) -> LogprobsTensors: ...
