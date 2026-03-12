import torch
from types import FunctionType
from vllm.config import ModelConfig

__all__ = ["set_torchao_reload_attrs", "support_quantized_model_reload_from_hp_weights"]

def set_torchao_reload_attrs(model: torch.nn.Module, model_config: ModelConfig): ...
def support_quantized_model_reload_from_hp_weights(
    original_load_weights: FunctionType,
): ...
