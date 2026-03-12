import torch.nn as nn
from _typeshed import Incomplete
from vllm.config import ModelConfig as ModelConfig, VllmConfig as VllmConfig
from vllm.config.load import LoadConfig as LoadConfig
from vllm.logger import init_logger as init_logger
from vllm.model_executor.model_loader.base_loader import (
    BaseModelLoader as BaseModelLoader,
)
from vllm.model_executor.model_loader.utils import (
    initialize_model as initialize_model,
    process_weights_after_loading as process_weights_after_loading,
)
from vllm.model_executor.model_loader.weight_utils import (
    download_gguf as download_gguf,
    get_gguf_extra_tensor_names as get_gguf_extra_tensor_names,
    get_gguf_weight_type_map as get_gguf_weight_type_map,
    gguf_quant_weights_iterator as gguf_quant_weights_iterator,
)
from vllm.transformers_utils.gguf_utils import (
    detect_gguf_multimodal as detect_gguf_multimodal,
)
from vllm.utils.torch_utils import set_default_torch_dtype as set_default_torch_dtype

logger: Incomplete

class GGUFModelLoader(BaseModelLoader):
    def __init__(self, load_config: LoadConfig) -> None: ...
    def download_model(self, model_config: ModelConfig) -> None: ...
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None: ...
    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module: ...
