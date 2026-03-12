import torch.nn as nn
from vllm.config import ModelConfig as ModelConfig
from vllm.config.load import LoadConfig as LoadConfig
from vllm.model_executor.model_loader.base_loader import (
    BaseModelLoader as BaseModelLoader,
)
from vllm.model_executor.model_loader.weight_utils import (
    initialize_dummy_weights as initialize_dummy_weights,
)

class DummyModelLoader(BaseModelLoader):
    def __init__(self, load_config: LoadConfig) -> None: ...
    def download_model(self, model_config: ModelConfig) -> None: ...
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None: ...
