import abc
import torch.nn as nn
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from vllm.config import ModelConfig as ModelConfig, VllmConfig as VllmConfig
from vllm.config.load import LoadConfig as LoadConfig
from vllm.logger import init_logger as init_logger
from vllm.model_executor.model_loader.utils import (
    initialize_model as initialize_model,
    process_weights_after_loading as process_weights_after_loading,
)
from vllm.platforms import current_platform as current_platform
from vllm.tracing import instrument as instrument
from vllm.utils.mem_utils import format_gib as format_gib
from vllm.utils.torch_utils import set_default_torch_dtype as set_default_torch_dtype

logger: Incomplete

class BaseModelLoader(ABC, metaclass=abc.ABCMeta):
    load_config: Incomplete
    def __init__(self, load_config: LoadConfig) -> None: ...
    @abstractmethod
    def download_model(self, model_config: ModelConfig) -> None: ...
    @abstractmethod
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None: ...
    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module: ...

def log_model_inspection(model: nn.Module) -> None: ...
