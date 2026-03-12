import torch
from vllm.config import VllmConfig as VllmConfig

def init_speculator(vllm_config: VllmConfig, device: torch.device): ...
