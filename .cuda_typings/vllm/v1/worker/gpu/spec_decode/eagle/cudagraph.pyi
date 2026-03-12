import torch
from _typeshed import Incomplete
from collections.abc import Callable as Callable
from vllm.config import VllmConfig as VllmConfig
from vllm.config.compilation import CUDAGraphMode as CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig as KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables as BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor as BatchExecutionDescriptor,
    CudaGraphManager as CudaGraphManager,
    prepare_inputs_to_capture as prepare_inputs_to_capture,
)
from vllm.v1.worker.gpu.input_batch import InputBuffers as InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState as ModelState
from vllm.v1.worker.utils import AttentionGroup as AttentionGroup

class EagleCudaGraphManager(CudaGraphManager):
    draft_tokens: Incomplete
    pool: Incomplete
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        draft_tokens: torch.Tensor,
    ) -> None: ...
    def capture(
        self,
        generate_fn: Callable,
        model_state: ModelState,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None: ...
    def run_fullgraph(self, desc: BatchExecutionDescriptor) -> torch.Tensor: ...
