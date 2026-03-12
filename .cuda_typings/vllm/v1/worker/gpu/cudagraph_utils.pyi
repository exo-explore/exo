import torch
import torch.nn as nn
from _typeshed import Incomplete
from collections.abc import Callable as Callable
from dataclasses import dataclass
from typing import Any
from vllm.config import VllmConfig as VllmConfig
from vllm.config.compilation import CUDAGraphMode as CUDAGraphMode
from vllm.distributed.parallel_state import (
    graph_capture as graph_capture,
    is_global_first_rank as is_global_first_rank,
)
from vllm.forward_context import (
    BatchDescriptor as BatchDescriptor,
    set_forward_context as set_forward_context,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.offloader.base import get_offloader as get_offloader
from vllm.platforms import current_platform as current_platform
from vllm.v1.kv_cache_interface import KVCacheConfig as KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import (
    build_slot_mappings_by_layer as build_slot_mappings_by_layer,
)
from vllm.v1.worker.gpu.block_table import BlockTables as BlockTables
from vllm.v1.worker.gpu.cp_utils import (
    prepare_dcp_local_seq_lens as prepare_dcp_local_seq_lens,
)
from vllm.v1.worker.gpu.input_batch import (
    InputBatch as InputBatch,
    InputBuffers as InputBuffers,
)
from vllm.v1.worker.gpu.model_states.interface import ModelState as ModelState
from vllm.v1.worker.utils import AttentionGroup as AttentionGroup

logger: Incomplete

@dataclass(frozen=True)
class BatchExecutionDescriptor:
    cg_mode: CUDAGraphMode
    num_tokens: int
    num_reqs: int | None
    uniform_token_count: int | None = ...

def get_uniform_token_count(
    num_reqs: int, num_tokens: int, max_query_len: int
) -> int | None: ...

class CudaGraphManager:
    vllm_config: Incomplete
    device: Incomplete
    max_num_reqs: Incomplete
    compilation_config: Incomplete
    cudagraph_mode: Incomplete
    decode_query_len: Incomplete
    dp_size: Incomplete
    graphs: dict[BatchExecutionDescriptor, torch.cuda.CUDAGraph]
    pool: Incomplete
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int,
    ) -> None: ...
    def needs_capture(self) -> bool: ...
    def capture(
        self,
        create_forward_fn: Callable[
            [BatchExecutionDescriptor], Callable[[CUDAGraphMode], None]
        ],
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None: ...
    def dispatch(
        self, num_reqs: int, num_tokens: int, uniform_token_count: int | None
    ) -> BatchExecutionDescriptor: ...
    def run_fullgraph(self, desc: BatchExecutionDescriptor): ...

class ModelCudaGraphManager(CudaGraphManager):
    hidden_states: torch.Tensor | None
    aux_hidden_states: list[torch.Tensor]
    use_aux_hidden_state_outputs: bool
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int,
    ) -> None: ...
    def capture(
        self,
        model: nn.Module,
        model_state: ModelState,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        has_lora: bool = False,
        use_aux_hidden_state_outputs: bool = False,
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None: ...
    def run_fullgraph(
        self, desc: BatchExecutionDescriptor
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]: ...

def prepare_inputs_to_capture(
    num_reqs: int,
    num_tokens: int,
    model_state: ModelState,
    input_buffers: InputBuffers,
    block_tables: BlockTables,
    attn_groups: list[list[AttentionGroup]],
    kv_cache_config: KVCacheConfig,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]: ...
