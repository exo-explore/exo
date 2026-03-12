import torch
from _typeshed import Incomplete
from collections.abc import Generator
from torch import nn as nn
from vllm.config import ModelConfig as ModelConfig
from vllm.config.load import LoadConfig as LoadConfig
from vllm.logger import init_logger as init_logger
from vllm.model_executor.model_loader.base_loader import (
    BaseModelLoader as BaseModelLoader,
)
from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf as download_weights_from_hf,
    runai_safetensors_weights_iterator as runai_safetensors_weights_iterator,
)
from vllm.transformers_utils.utils import is_s3 as is_s3

logger: Incomplete

class ShardedStateLoader(BaseModelLoader):
    DEFAULT_PATTERN: str
    pattern: Incomplete
    def __init__(self, load_config: LoadConfig) -> None: ...
    def download_model(self, model_config: ModelConfig) -> None: ...
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None: ...
    def iterate_over_files(
        self, paths
    ) -> Generator[tuple[str, torch.Tensor], None, None]: ...
    @staticmethod
    def save_model(
        model: torch.nn.Module,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None: ...
