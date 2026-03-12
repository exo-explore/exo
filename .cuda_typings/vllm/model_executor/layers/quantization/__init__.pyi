from _typeshed import Incomplete
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig as QuantizationConfig,
)

__all__ = [
    "QuantizationConfig",
    "QuantizationMethods",
    "get_quantization_config",
    "register_quantization_config",
    "QUANTIZATION_METHODS",
]

QuantizationMethods: Incomplete
QUANTIZATION_METHODS: list[str]

def register_quantization_config(quantization: str): ...
def get_quantization_config(quantization: str) -> type[QuantizationConfig]: ...
