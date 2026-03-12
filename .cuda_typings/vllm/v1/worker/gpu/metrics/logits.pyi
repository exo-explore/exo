import torch
from vllm.triton_utils import tl as tl, triton as triton

def get_num_nans(logits: torch.Tensor) -> torch.Tensor: ...
