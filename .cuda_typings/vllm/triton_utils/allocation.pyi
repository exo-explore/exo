import torch
from vllm.triton_utils import triton as triton

def set_triton_allocator(device: torch.device): ...
