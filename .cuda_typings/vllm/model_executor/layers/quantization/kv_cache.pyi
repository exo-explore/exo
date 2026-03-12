import torch
from _typeshed import Incomplete
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig as QuantizationConfig,
    QuantizeMethodBase as QuantizeMethodBase,
)
from vllm.platforms import current_platform as current_platform
from vllm.v1.attention.backend import is_quantized_kv_cache as is_quantized_kv_cache

logger: Incomplete

class BaseKVCacheMethod(QuantizeMethodBase):
    quant_config: Incomplete
    def __init__(self, quant_config: QuantizationConfig) -> None: ...
    def create_weights(self, layer: torch.nn.Module): ...
    def apply(self, layer: torch.nn.Module) -> torch.Tensor: ...
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None: ...
