import abc
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any
from vllm.config import VllmConfig as VllmConfig
from vllm.config.compilation import CUDAGraphMode as CUDAGraphMode
from vllm.v1.core.sched.output import NewRequestData as NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig as KVCacheConfig
from vllm.v1.worker.gpu.input_batch import InputBatch as InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache as EncoderCache
from vllm.v1.worker.gpu.states import RequestState as RequestState
from vllm.v1.worker.utils import AttentionGroup as AttentionGroup

class ModelState(ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ): ...
    @abstractmethod
    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None: ...
    @abstractmethod
    def apply_staged_writes(self) -> None: ...
    @abstractmethod
    def get_mm_embeddings(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor: ...
    @abstractmethod
    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, torch.Tensor | None]: ...
    @abstractmethod
    def prepare_dummy_inputs(
        self, num_reqs: int, num_tokens: int
    ) -> dict[str, torch.Tensor | None]: ...
    @abstractmethod
    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
    ) -> dict[str, Any]: ...
