import torch.nn as nn
from vllm.config import VllmConfig as VllmConfig
from vllm.model_executor.model_loader import get_model as get_model

def load_eagle_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module: ...
