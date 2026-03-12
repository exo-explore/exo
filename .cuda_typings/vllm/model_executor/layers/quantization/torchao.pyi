import torch
from _typeshed import Incomplete
from typing import Any
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.linear import (
    LinearBase as LinearBase,
    LinearMethodBase as LinearMethodBase,
    UnquantizedLinearMethod as UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import (
    QuantizationMethods as QuantizationMethods,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig as QuantizationConfig,
    QuantizeMethodBase as QuantizeMethodBase,
)
from vllm.model_executor.utils import set_weight_attrs as set_weight_attrs

logger: Incomplete

def torchao_version_at_least(torchao_version: str) -> bool: ...
def should_skip(prefix: str, skip_modules: list[str]) -> bool: ...

class TorchAOConfig(QuantizationConfig):
    torchao_config: Incomplete
    skip_modules: Incomplete
    is_checkpoint_torchao_serialized: Incomplete
    def __init__(
        self,
        torchao_config,
        skip_modules: list[str] | None = None,
        is_checkpoint_torchao_serialized: bool = False,
    ) -> None: ...
    def get_name(self) -> QuantizationMethods: ...
    def get_supported_act_dtypes(self) -> list[torch.dtype]: ...
    @classmethod
    def get_min_capability(cls) -> int: ...
    @staticmethod
    def get_config_filenames() -> list[str]: ...
    @classmethod
    def from_config(cls, config: dict[str, Any]) -> TorchAOConfig: ...
    @classmethod
    def from_config_file(cls, config_file: str) -> TorchAOConfig: ...
    @classmethod
    def from_config_dict_json(cls, config_dict_json: str) -> TorchAOConfig: ...
    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None: ...
    def get_scaled_act_names(self) -> list[str]: ...

def torchao_quantize_param_data(
    param: torch.Tensor, torchao_config: Any
) -> torch.nn.Parameter: ...

class TorchAOLinearMethod(LinearMethodBase):
    quant_config: Incomplete
    def __init__(self, quant_config: TorchAOConfig) -> None: ...
    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ): ...
    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor: ...
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None: ...
