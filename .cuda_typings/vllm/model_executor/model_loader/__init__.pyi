from torch import nn
from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.base_loader import (
    BaseModelLoader as BaseModelLoader,
)
from vllm.model_executor.model_loader.bitsandbytes_loader import (
    BitsAndBytesModelLoader as BitsAndBytesModelLoader,
)
from vllm.model_executor.model_loader.default_loader import (
    DefaultModelLoader as DefaultModelLoader,
)
from vllm.model_executor.model_loader.dummy_loader import (
    DummyModelLoader as DummyModelLoader,
)
from vllm.model_executor.model_loader.gguf_loader import (
    GGUFModelLoader as GGUFModelLoader,
)
from vllm.model_executor.model_loader.runai_streamer_loader import (
    RunaiModelStreamerLoader as RunaiModelStreamerLoader,
)
from vllm.model_executor.model_loader.sharded_state_loader import (
    ShardedStateLoader as ShardedStateLoader,
)
from vllm.model_executor.model_loader.tensorizer_loader import (
    TensorizerLoader as TensorizerLoader,
)
from vllm.model_executor.model_loader.utils import (
    get_architecture_class_name as get_architecture_class_name,
    get_model_architecture as get_model_architecture,
    get_model_cls as get_model_cls,
)

__all__ = [
    "get_model",
    "get_model_loader",
    "get_architecture_class_name",
    "get_model_architecture",
    "get_model_cls",
    "register_model_loader",
    "BaseModelLoader",
    "BitsAndBytesModelLoader",
    "GGUFModelLoader",
    "DefaultModelLoader",
    "DummyModelLoader",
    "RunaiModelStreamerLoader",
    "ShardedStateLoader",
    "TensorizerLoader",
]

def register_model_loader(load_format: str): ...
def get_model_loader(load_config: LoadConfig) -> BaseModelLoader: ...
def get_model(
    *,
    vllm_config: VllmConfig,
    model_config: ModelConfig | None = None,
    prefix: str = "",
    load_config: LoadConfig | None = None,
) -> nn.Module: ...
