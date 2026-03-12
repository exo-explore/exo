import torch
from _typeshed import Incomplete
from collections.abc import Callable as Callable
from torch import nn as nn
from typing import TypeVar
from vllm.config import VllmConfig as VllmConfig
from vllm.config.lora import LoRAConfig as LoRAConfig
from vllm.logger import init_logger as init_logger
from vllm.lora.layers import (
    BaseLayerWithLoRA as BaseLayerWithLoRA,
    FusedMoE3DWithLoRA as FusedMoE3DWithLoRA,
    LoRAMapping as LoRAMapping,
    LoRAMappingType as LoRAMappingType,
)
from vllm.lora.lora_model import LoRAModel as LoRAModel
from vllm.lora.lora_weights import (
    LoRALayerWeights as LoRALayerWeights,
    PackedLoRALayerWeights as PackedLoRALayerWeights,
)
from vllm.lora.punica_wrapper import (
    PunicaWrapperBase as PunicaWrapperBase,
    get_punica_wrapper as get_punica_wrapper,
)
from vllm.lora.utils import (
    from_layer as from_layer,
    from_layer_logits_processor as from_layer_logits_processor,
    get_supported_lora_modules as get_supported_lora_modules,
    is_moe_model as is_moe_model,
    process_packed_modules_mapping as process_packed_modules_mapping,
    replace_submodule as replace_submodule,
)
from vllm.model_executor.layers.fused_moe import FusedMoE as FusedMoE
from vllm.model_executor.models import (
    SupportsLoRA as SupportsLoRA,
    is_pooling_model as is_pooling_model,
    supports_multimodal as supports_multimodal,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys as MultiModelKeys
from vllm.model_executor.models.utils import PPMissingLayer as PPMissingLayer
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.encoder_budget import MultiModalBudget as MultiModalBudget
from vllm.utils.cache import LRUCache as LRUCache
from vllm.utils.platform_utils import is_pin_memory_available as is_pin_memory_available

logger: Incomplete
T = TypeVar("T")
DEFAULT_LANGUAGE_WRAPPER_KEY: str

class AdapterLRUCache(LRUCache[int, T]):
    deactivate_fn: Incomplete
    def __init__(
        self, capacity: int, deactivate_fn: Callable[[int], object]
    ) -> None: ...

class LoRAModelManager:
    model: SupportsLoRA
    supported_lora_modules: Incomplete
    adapter_type: str
    lora_config: Incomplete
    device: Incomplete
    max_num_seqs: Incomplete
    max_num_batched_tokens: Incomplete
    lora_index_to_id: list[int | None]
    vocab_size: Incomplete
    packed_modules_mapping: Incomplete
    is_pooling_model: Incomplete
    packed_modules: dict[str, list[str]]
    modules: dict[str, BaseLayerWithLoRA]
    def __init__(
        self,
        model: SupportsLoRA,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        vocab_size: int,
        lora_config: LoRAConfig,
        device: torch.device,
        vllm_config: VllmConfig | None = None,
    ) -> None: ...
    def __len__(self) -> int: ...
    @property
    def capacity(self) -> int: ...
    @property
    def lora_slots(self) -> int: ...
    @property
    def adapter_slots(self) -> int: ...
    def activate_adapter(self, lora_id: int) -> bool: ...
    def pin_adapter(self, lora_id: int) -> bool: ...
    def remove_all_adapters(self) -> None: ...
    def register_module(self, module_name: str, module: BaseLayerWithLoRA): ...
    def create_dummy_lora(
        self, lora_id: int, rank: int, embedding_modules: dict[str, str] | None = None
    ) -> LoRAModel: ...
    def deactivate_adapter(self, adapter_id: int) -> bool: ...
    def add_adapter(self, adapter: LoRAModel) -> bool: ...
    def set_adapter_mapping(self, mapping: LoRAMapping) -> None: ...
    def remove_adapter(self, adapter_id: int) -> bool: ...
    def list_adapters(self) -> dict[int, LoRAModel]: ...
    def get_adapter(self, adapter_id: int) -> LoRAModel | None: ...

class LoRALRUCache(AdapterLRUCache[LoRAModel]):
    def __init__(
        self, capacity: int, deactivate_lora_fn: Callable[[int], bool]
    ) -> None: ...

class LRUCacheLoRAModelManager(LoRAModelManager):
    def __init__(
        self,
        model: nn.Module,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        vocab_size: int,
        lora_config: LoRAConfig,
        device: torch.device,
        vllm_config: VllmConfig | None = None,
    ) -> None: ...
    def list_adapters(self) -> dict[int, LoRAModel]: ...
    def add_adapter(self, lora: LoRAModel) -> bool: ...
    def activate_adapter(self, lora_id: int) -> bool: ...
    def remove_oldest_adapter(self) -> bool: ...
    def pin_adapter(self, lora_id: int) -> bool: ...

def create_lora_manager(
    model: nn.Module,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    vocab_size: int,
    lora_config: LoRAConfig,
    vllm_config: VllmConfig,
    device: torch.device,
    lora_manager_cls: type[LoRAModelManager] = ...,
    **kwargs,
) -> LoRAModelManager: ...
