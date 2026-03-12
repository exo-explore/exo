import torch
from _typeshed import Incomplete
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from vllm.config import VllmConfig as VllmConfig
from vllm.exceptions import LoRAAdapterNotFoundError as LoRAAdapterNotFoundError
from vllm.logger import init_logger as init_logger
from vllm.lora.lora_model import LoRAModel as LoRAModel
from vllm.lora.model_manager import (
    LRUCacheLoRAModelManager as LRUCacheLoRAModelManager,
    LoRAModelManager as LoRAModelManager,
    create_lora_manager as create_lora_manager,
)
from vllm.lora.peft_helper import PEFTHelper as PEFTHelper
from vllm.lora.request import LoRARequest as LoRARequest
from vllm.lora.utils import get_adapter_absolute_path as get_adapter_absolute_path

logger: Incomplete

class WorkerLoRAManager:
    embedding_modules: Incomplete
    max_num_seqs: Incomplete
    max_num_batched_tokens: Incomplete
    vocab_size: Incomplete
    lora_config: Incomplete
    max_position_embeddings: Incomplete
    device: Incomplete
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        embedding_modules: dict[str, str],
        lora_model_cls: type[LoRAModel] = ...,
    ) -> None: ...
    @contextmanager
    def dummy_lora_cache(self) -> Generator[None]: ...
    @property
    def is_enabled(self) -> bool: ...
    def create_lora_manager(
        self, model: torch.nn.Module, vllm_config: VllmConfig | None = None
    ) -> Any: ...
    def add_dummy_lora(self, lora_request: LoRARequest, rank: int) -> bool: ...
    def pin_adapter(self, adapter_id: int) -> bool: ...
    def set_active_adapters(self, requests: set[Any], mapping: Any | None) -> None: ...
    def supports_tower_connector_lora(self) -> bool: ...
    def add_adapter(self, adapter_request: Any) -> bool: ...
    def remove_adapter(self, adapter_id: int) -> bool: ...
    def remove_all_adapters(self) -> None: ...
    def list_adapters(self) -> set[int]: ...

class LRUCacheWorkerLoRAManager(WorkerLoRAManager):
    def create_lora_manager(
        self, model: torch.nn.Module, vllm_config: VllmConfig | None = None
    ) -> Any: ...
    def add_adapter(self, lora_request: LoRARequest) -> bool: ...
