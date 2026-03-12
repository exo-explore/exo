from torch import nn as nn
from vllm.config import ModelConfig as ModelConfig
from vllm.config.load import LoadConfig as LoadConfig
from vllm.model_executor.model_loader.base_loader import (
    BaseModelLoader as BaseModelLoader,
)
from vllm.model_executor.model_loader.weight_utils import (
    download_safetensors_index_file_from_hf as download_safetensors_index_file_from_hf,
    download_weights_from_hf as download_weights_from_hf,
    runai_safetensors_weights_iterator as runai_safetensors_weights_iterator,
)
from vllm.transformers_utils.runai_utils import (
    is_runai_obj_uri as is_runai_obj_uri,
    list_safetensors as list_safetensors,
)

class RunaiModelStreamerLoader(BaseModelLoader):
    def __init__(self, load_config: LoadConfig) -> None: ...
    def download_model(self, model_config: ModelConfig) -> None: ...
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None: ...
