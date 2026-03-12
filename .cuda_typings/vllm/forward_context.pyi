import torch
from _typeshed import Incomplete
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from vllm.config import (
    CUDAGraphMode as CUDAGraphMode,
    ParallelConfig as ParallelConfig,
    VllmConfig as VllmConfig,
)
from vllm.logger import init_logger as init_logger
from vllm.platforms import current_platform as current_platform
from vllm.v1.attention.backend import AttentionMetadata as AttentionMetadata
from vllm.v1.worker.dp_utils import (
    coordinate_batch_across_dp as coordinate_batch_across_dp,
)
from vllm.v1.worker.ubatch_utils import UBatchSlices as UBatchSlices

logger: Incomplete
track_batchsize: bool
last_logging_time: float
forward_start_time: float
batchsize_logging_interval: float
batchsize_forward_time: defaultdict

@dataclass(frozen=True)
class BatchDescriptor:
    num_tokens: int
    num_reqs: int | None = ...
    uniform: bool = ...
    has_lora: bool = ...
    num_active_loras: int = ...

@dataclass
class DPMetadata:
    max_tokens_across_dp_cpu: torch.Tensor
    num_tokens_across_dp_cpu: torch.Tensor
    local_sizes: list[int] | None = ...
    @staticmethod
    def make(
        parallel_config: ParallelConfig,
        num_tokens: int,
        num_tokens_across_dp_cpu: torch.Tensor,
    ) -> DPMetadata: ...
    @contextmanager
    def chunked_sizes(
        self, sequence_parallel_size: int, max_chunk_size_per_rank: int, chunk_idx: int
    ): ...
    @contextmanager
    def sp_local_sizes(self, sequence_parallel_size: int): ...
    def get_chunk_sizes_across_dp_rank(self) -> list[int] | None: ...
    def cu_tokens_across_sp(self, sp_size: int) -> torch.Tensor: ...

@dataclass
class ForwardContext:
    no_compile_layers: dict[str, Any]
    attn_metadata: dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]]
    slot_mapping: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]
    virtual_engine: int
    dp_metadata: DPMetadata | None = ...
    cudagraph_runtime_mode: CUDAGraphMode = ...
    batch_descriptor: BatchDescriptor | None = ...
    ubatch_slices: UBatchSlices | None = ...
    skip_compiled: bool = ...
    all_moe_layers: list[str] | None = ...
    moe_layer_index: int = ...
    additional_kwargs: dict[str, Any] = field(default_factory=dict)
    def __post_init__(self) -> None: ...

def get_forward_context() -> ForwardContext: ...
def is_forward_context_available() -> bool: ...
def create_forward_context(
    attn_metadata: Any,
    vllm_config: VllmConfig,
    virtual_engine: int = 0,
    dp_metadata: DPMetadata | None = None,
    cudagraph_runtime_mode: CUDAGraphMode = ...,
    batch_descriptor: BatchDescriptor | None = None,
    ubatch_slices: UBatchSlices | None = None,
    slot_mapping: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None,
    additional_kwargs: dict[str, Any] | None = None,
    skip_compiled: bool = False,
): ...
@contextmanager
def override_forward_context(forward_context: ForwardContext | None): ...
@contextmanager
def set_forward_context(
    attn_metadata: Any,
    vllm_config: VllmConfig,
    virtual_engine: int = 0,
    num_tokens: int | None = None,
    num_tokens_across_dp: torch.Tensor | None = None,
    cudagraph_runtime_mode: CUDAGraphMode = ...,
    batch_descriptor: BatchDescriptor | None = None,
    ubatch_slices: UBatchSlices | None = None,
    slot_mapping: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None,
    skip_compiled: bool = False,
): ...
