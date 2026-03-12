import torch
from _typeshed import Incomplete
from contextlib import contextmanager
from dataclasses import dataclass, field
from torch import nn
from vllm.config import (
    ModelConfig as ModelConfig,
    VllmConfig as VllmConfig,
    set_current_vllm_config as set_current_vllm_config,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.attention import (
    Attention as Attention,
    MLAAttention as MLAAttention,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig as QuantizationConfig,
    QuantizeMethodBase as QuantizeMethodBase,
)
from vllm.model_executor.model_loader.reload import (
    record_metadata_for_reloading as record_metadata_for_reloading,
    set_torchao_reload_attrs as set_torchao_reload_attrs,
)
from vllm.model_executor.models.interfaces import SupportsQuant as SupportsQuant
from vllm.tracing import instrument as instrument
from vllm.utils.platform_utils import is_pin_memory_available as is_pin_memory_available
from vllm.utils.torch_utils import (
    get_accelerator_view_from_cpu_tensor as get_accelerator_view_from_cpu_tensor,
)

logger: Incomplete

def initialize_model(
    vllm_config: VllmConfig,
    *,
    prefix: str = "",
    model_class: type[nn.Module] | None = None,
    model_config: ModelConfig | None = None,
) -> nn.Module: ...
def process_weights_after_loading(
    model: nn.Module, model_config: ModelConfig, target_device: torch.device
) -> None: ...
@contextmanager
def device_loading_context(module: torch.nn.Module, target_device: torch.device): ...
def get_model_architecture(
    model_config: ModelConfig,
) -> tuple[type[nn.Module], str]: ...
def get_model_cls(model_config: ModelConfig) -> type[nn.Module]: ...
def get_architecture_class_name(model_config: ModelConfig) -> str: ...
@dataclass
class ParamMapping:
    packed_mapping: dict[str, list[str]]
    inverse_packed_mapping: dict[str, tuple[str, int]] = field(default_factory=dict)
    def __post_init__(self) -> None: ...
    def get_sub_modules(self, module_name: str) -> tuple[str, list[str]] | None: ...

def configure_quant_config(
    quant_config: QuantizationConfig, model_class: type[nn.Module]
): ...
