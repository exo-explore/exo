import torch
import torch.nn as nn
from _typeshed import Incomplete
from typing import Any
from vllm.config import VllmConfig as VllmConfig
from vllm.config.compilation import CUDAGraphMode as CUDAGraphMode
from vllm.v1.core.sched.output import NewRequestData as NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig as KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata as build_attn_metadata
from vllm.v1.worker.gpu.input_batch import InputBatch as InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache as EncoderCache
from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner as EncoderRunner
from vllm.v1.worker.gpu.mm.mrope_utils import MRopeState as MRopeState
from vllm.v1.worker.gpu.model_states.interface import ModelState as ModelState
from vllm.v1.worker.gpu.states import RequestState as RequestState
from vllm.v1.worker.utils import AttentionGroup as AttentionGroup

class DefaultModelState(ModelState):
    vllm_config: Incomplete
    model_config: Incomplete
    scheduler_config: Incomplete
    model: Incomplete
    device: Incomplete
    supports_mm_inputs: Incomplete
    max_model_len: Incomplete
    max_num_reqs: Incomplete
    max_num_tokens: Incomplete
    inputs_embeds_size: Incomplete
    dtype: Incomplete
    encoder_cache: Incomplete
    encoder_runner: Incomplete
    uses_mrope: Incomplete
    mrope_state: Incomplete
    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None: ...
    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None: ...
    def apply_staged_writes(self) -> None: ...
    def get_mm_embeddings(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor: ...
    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, torch.Tensor | None]: ...
    def prepare_dummy_inputs(
        self, num_reqs: int, num_tokens: int
    ) -> dict[str, torch.Tensor | None]: ...
    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
    ) -> dict[str, Any]: ...
