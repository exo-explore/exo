import torch
import torch.nn as nn
from _typeshed import Incomplete
from collections.abc import Callable as Callable, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from transformers import PretrainedConfig as PretrainedConfig
from typing import Any, Literal, Protocol, overload
from vllm.config import VllmConfig as VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.model_loader.reload import (
    support_quantized_model_reload_from_hp_weights as support_quantized_model_reload_from_hp_weights,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)
from vllm.model_executor.models.interfaces import (
    supports_any_eagle as supports_any_eagle,
)
from vllm.multimodal import NestedTensors as NestedTensors
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.utils.math_utils import cdiv as cdiv
from vllm.utils.platform_utils import is_pin_memory_available as is_pin_memory_available
from vllm.utils.torch_utils import (
    direct_register_custom_op as direct_register_custom_op,
)

logger: Incomplete
WeightsMapping = Mapping[str, str | None]

@dataclass
class WeightsMapper:
    orig_to_new_substr: WeightsMapping = field(default_factory=dict)
    orig_to_new_prefix: WeightsMapping = field(default_factory=dict)
    orig_to_new_suffix: WeightsMapping = field(default_factory=dict)
    def __or__(self, other: WeightsMapper) -> WeightsMapper: ...
    def apply(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]: ...
    def apply_list(self, values: list[str]) -> list[str]: ...
    def apply_dict(self, values: dict[str, Any]) -> dict[str, Any]: ...

class AutoWeightsLoader:
    ROTARY_EMBEDS_UNUSED_WEIGHTS: Incomplete
    module: Incomplete
    skip_prefixes: Incomplete
    skip_substrs: Incomplete
    ignore_unexpected_prefixes: Incomplete
    ignore_unexpected_suffixes: Incomplete
    def __init__(
        self,
        module: nn.Module,
        *,
        skip_prefixes: list[str] | None = None,
        skip_substrs: list[str] | None = None,
        ignore_unexpected_prefixes: list[str] | None = None,
        ignore_unexpected_suffixes: list[str] | None = None,
    ) -> None: ...
    @support_quantized_model_reload_from_hp_weights
    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        *,
        mapper: WeightsMapper | None = None,
    ) -> set[str]: ...

def init_vllm_registered_model(
    vllm_config: VllmConfig,
    *,
    prefix: str = "",
    hf_config: PretrainedConfig | None = None,
    architectures: list[str] | None = None,
) -> nn.Module: ...
@overload
def flatten_bn(x: torch.Tensor) -> torch.Tensor: ...
@overload
def flatten_bn(x: list[torch.Tensor]) -> list[torch.Tensor]: ...
@overload
def flatten_bn(
    x: list[torch.Tensor] | torch.Tensor, *, concat: Literal[True]
) -> torch.Tensor: ...
@overload
def flatten_bn(
    x: list[torch.Tensor] | torch.Tensor, *, concat: bool = False
) -> list[torch.Tensor] | torch.Tensor: ...
def split_list_into_ranges(lst: torch.Tensor, interval: int) -> list[list[int]]: ...
def isin_list(
    elements: torch.Tensor, test_elements_list: list[int]
) -> torch.Tensor: ...

class StageMissingLayer(nn.Module):
    stage_name: Incomplete
    def __init__(self, stage_name: str, module: nn.Module | None = None) -> None: ...
    def __getattr__(self, name: str): ...
    def __call__(self, *args, **kwargs) -> None: ...
    def extra_repr(self) -> str: ...

@contextmanager
def collect_children(
    module: nn.Module,
    *,
    targets: type[nn.Module] | tuple[type[nn.Module], ...] | None = None,
): ...
@contextmanager
def no_init_weights(
    module: nn.Module,
    placeholder: Callable[[nn.Module], nn.Module],
    *,
    targets: type[nn.Module] | tuple[type[nn.Module], ...] | None = None,
): ...

class LayerFn(Protocol):
    def __call__(self, prefix: str) -> torch.nn.Module: ...

class PPMissingLayer(torch.nn.Identity):
    def __init__(self, *args, **kwargs) -> None: ...
    def forward(self, *args, **kwargs): ...

def make_layers(
    num_hidden_layers: int, layer_fn: LayerFn, prefix: str
) -> tuple[int, int, torch.nn.ModuleList]: ...
def get_pp_missing_layer_names(model: torch.nn.Module) -> list[str]: ...
def is_pp_missing_parameter(name: str, model: torch.nn.Module) -> bool: ...
def make_empty_intermediate_tensors_factory(keys: list[str], hidden_size: int): ...
def maybe_prefix(prefix: str, name: str) -> str: ...
def get_draft_quant_config(vllm_config: VllmConfig) -> QuantizationConfig | None: ...
def extract_layer_index(layer_name: str, num_attn_module: int = 1) -> int: ...
def cast_overflow_tensors(
    tensors: torch.Tensor, offset: float = 1000
) -> torch.Tensor: ...
def fast_topk(
    values: torch.Tensor, topk: int, dim: int
) -> tuple[torch.Tensor, torch.Tensor]: ...
def sequence_parallel_chunk(x: torch.Tensor) -> torch.Tensor: ...
def sequence_parallel_chunk_impl(x: torch.Tensor) -> torch.Tensor: ...
def sequence_parallel_chunk_impl_fake(x: torch.Tensor) -> torch.Tensor: ...
def process_eagle_weight(model: nn.Module, name: str) -> None: ...
def get_layer_index(feature_layer_index: int, num_hidden_layers: int) -> int: ...
