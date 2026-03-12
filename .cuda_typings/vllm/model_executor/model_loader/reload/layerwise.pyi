import torch
from .types import LayerReloadingInfo
from vllm.config import ModelConfig

__all__ = [
    "get_layerwise_info",
    "record_metadata_for_reloading",
    "initialize_layerwise_reload",
    "finalize_layerwise_reload",
]

def get_layerwise_info(layer: torch.nn.Module) -> LayerReloadingInfo: ...
def record_metadata_for_reloading(model: torch.nn.Module): ...
def initialize_layerwise_reload(model: torch.nn.Module): ...
def finalize_layerwise_reload(model: torch.nn.Module, model_config: ModelConfig): ...
