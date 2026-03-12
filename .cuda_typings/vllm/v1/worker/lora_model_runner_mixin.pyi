import numpy as np
import torch
import torch.nn as nn
from _typeshed import Incomplete
from contextlib import contextmanager
from typing import TypeAlias
from vllm.config import VllmConfig as VllmConfig
from vllm.config.lora import LoRAConfig as LoRAConfig
from vllm.logger import init_logger as init_logger
from vllm.lora.layers import (
    LoRAMapping as LoRAMapping,
    LoRAMappingType as LoRAMappingType,
)
from vllm.lora.request import LoRARequest as LoRARequest
from vllm.lora.worker_manager import (
    LRUCacheWorkerLoRAManager as LRUCacheWorkerLoRAManager,
)
from vllm.model_executor.models import supports_lora as supports_lora
from vllm.v1.worker.gpu_input_batch import InputBatch as GPUInputBatch
from vllm.v1.worker.tpu_input_batch import InputBatch as TPUInputBatch

InputBatch: TypeAlias = TPUInputBatch | GPUInputBatch
logger: Incomplete

class LoRAModelRunnerMixin:
    lora_manager: Incomplete
    def load_lora_model(
        self, model: nn.Module, vllm_config: VllmConfig, device: torch.device
    ) -> nn.Module: ...
    def set_active_loras(
        self,
        input_batch: InputBatch,
        num_scheduled_tokens: np.ndarray,
        num_sampled_tokens: np.ndarray | None = None,
        mapping_type: LoRAMappingType = ...,
    ) -> None: ...
    @contextmanager
    def maybe_setup_dummy_loras(
        self, lora_config: LoRAConfig | None, remove_lora: bool = True
    ): ...
    @contextmanager
    def maybe_select_dummy_loras(
        self,
        lora_config: LoRAConfig | None,
        num_scheduled_tokens: np.ndarray,
        mapping_type: LoRAMappingType = ...,
        num_sampled_tokens: np.ndarray | None = None,
        num_active_loras: int = 0,
    ): ...
    @contextmanager
    def maybe_dummy_run_with_lora(
        self,
        lora_config: LoRAConfig | None,
        num_scheduled_tokens: np.ndarray,
        num_sampled_tokens: np.ndarray,
        remove_lora: bool = True,
        num_active_loras: int = 0,
        mapping_type: LoRAMappingType = ...,
    ): ...
    def maybe_remove_all_loras(self, lora_config: LoRAConfig | None): ...
    def add_lora(self, lora_request: LoRARequest) -> bool: ...
    def remove_lora(self, lora_id: int) -> bool: ...
    def pin_lora(self, lora_id: int) -> bool: ...
    def list_loras(self) -> set[int]: ...
