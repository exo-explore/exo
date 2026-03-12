import torch
import torch.nn as nn
from _typeshed import Incomplete
from typing import Any
from vllm.config import VllmConfig as VllmConfig
from vllm.config.compilation import CUDAGraphMode as CUDAGraphMode
from vllm.forward_context import (
    BatchDescriptor as BatchDescriptor,
    set_forward_context as set_forward_context,
)
from vllm.logger import init_logger as init_logger
from vllm.triton_utils import tl as tl, triton as triton
from vllm.v1.kv_cache_interface import KVCacheConfig as KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata as build_attn_metadata,
    build_slot_mappings_by_layer as build_slot_mappings_by_layer,
)
from vllm.v1.worker.gpu.block_table import BlockTables as BlockTables
from vllm.v1.worker.gpu.dp_utils import (
    sync_cudagraph_and_dp_padding as sync_cudagraph_and_dp_padding,
)
from vllm.v1.worker.gpu.input_batch import (
    InputBatch as InputBatch,
    InputBuffers as InputBuffers,
)
from vllm.v1.worker.gpu.model_states.interface import ModelState as ModelState
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample as gumbel_sample
from vllm.v1.worker.gpu.spec_decode.eagle.cudagraph import (
    EagleCudaGraphManager as EagleCudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
    load_eagle_model as load_eagle_model,
)
from vllm.v1.worker.utils import AttentionGroup as AttentionGroup

logger: Incomplete

class EagleSpeculator:
    vllm_config: Incomplete
    device: Incomplete
    speculative_config: Incomplete
    method: Incomplete
    num_speculative_steps: Incomplete
    draft_model_config: Incomplete
    scheduler_config: Incomplete
    max_num_reqs: Incomplete
    max_num_tokens: Incomplete
    max_model_len: Incomplete
    hidden_size: Incomplete
    vocab_size: Incomplete
    dtype: Incomplete
    dp_size: Incomplete
    dp_rank: Incomplete
    input_buffers: Incomplete
    hidden_states: Incomplete
    idx_mapping: Incomplete
    temperature: Incomplete
    seeds: Incomplete
    draft_tokens: Incomplete
    cudagraph_manager: Incomplete
    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None: ...
    model: Incomplete
    def load_model(self, target_model: nn.Module) -> None: ...
    model_state: Incomplete
    kv_cache_config: Incomplete
    attn_groups: Incomplete
    block_tables: Incomplete
    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        attn_groups: list[list[AttentionGroup]],
        block_tables: BlockTables,
    ) -> None: ...
    def run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = ...,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = ...,
    ) -> None: ...
    def capture_model(self) -> None: ...
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
    ) -> torch.Tensor: ...

def prepare_eagle_inputs(
    input_buffers: InputBuffers,
    input_batch: InputBatch,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    last_sampled: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
) -> torch.Tensor: ...
def prepare_eagle_decode(
    draft_tokens: torch.Tensor,
    output_hidden_states: torch.Tensor,
    last_token_indices: torch.Tensor,
    target_seq_lens: torch.Tensor,
    num_rejected: torch.Tensor,
    input_buffers: InputBuffers,
    input_hidden_states: torch.Tensor,
    max_model_len: int,
    max_num_reqs: int,
): ...
def update_eagle_inputs(
    draft_tokens: torch.Tensor,
    output_hidden_states: torch.Tensor,
    input_buffers: InputBuffers,
    hidden_states: torch.Tensor,
    max_model_len: int,
): ...
