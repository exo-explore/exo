import torch
from _typeshed import Incomplete
from torch import nn as nn
from vllm.config import (
    ModelConfig as ModelConfig,
    ParallelConfig as ParallelConfig,
    VllmConfig as VllmConfig,
)
from vllm.config.load import LoadConfig as LoadConfig
from vllm.logger import init_logger as init_logger
from vllm.model_executor.model_loader.base_loader import (
    BaseModelLoader as BaseModelLoader,
)
from vllm.model_executor.model_loader.tensorizer import (
    TensorizerConfig as TensorizerConfig,
    deserialize_tensorizer_model as deserialize_tensorizer_model,
    init_tensorizer_model as init_tensorizer_model,
    is_vllm_tensorized as is_vllm_tensorized,
    serialize_vllm_model as serialize_vllm_model,
    tensorizer_weights_iterator as tensorizer_weights_iterator,
)
from vllm.model_executor.model_loader.utils import (
    get_model_architecture as get_model_architecture,
    initialize_model as initialize_model,
)
from vllm.utils.torch_utils import set_default_torch_dtype as set_default_torch_dtype

logger: Incomplete
BLACKLISTED_TENSORIZER_ARGS: Incomplete

def validate_config(config: dict): ...

class TensorizerLoader(BaseModelLoader):
    tensorizer_config: Incomplete
    def __init__(self, load_config: LoadConfig) -> None: ...
    def download_model(self, model_config: ModelConfig) -> None: ...
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None: ...
    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module: ...
    @staticmethod
    def save_model(
        model: torch.nn.Module,
        tensorizer_config: TensorizerConfig | dict,
        model_config: ModelConfig,
    ) -> None: ...
