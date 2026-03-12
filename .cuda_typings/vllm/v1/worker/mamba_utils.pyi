import dataclasses
from collections.abc import Callable as Callable
from typing import Any
from vllm.config import CacheConfig as CacheConfig
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc as MambaStateCopyFunc,
)
from vllm.triton_utils import tl as tl, triton as triton
from vllm.utils.math_utils import cdiv as cdiv
from vllm.v1.core.sched.output import SchedulerOutput as SchedulerOutput
from vllm.v1.kv_cache_interface import (
    KVCacheConfig as KVCacheConfig,
    MambaSpec as MambaSpec,
)
from vllm.v1.utils import CpuGpuBuffer as CpuGpuBuffer
from vllm.v1.worker.gpu_input_batch import CachedRequestState as CachedRequestState
from vllm.v1.worker.lora_model_runner_mixin import GPUInputBatch as GPUInputBatch

@triton.jit
def batch_memcpy_kernel(src_ptrs, dst_ptrs, sizes, BLOCK_SIZE: tl.constexpr): ...
def batch_memcpy(src_ptrs, dst_ptrs, sizes) -> None: ...
def get_mamba_groups(kv_cache_config: KVCacheConfig) -> tuple[list[int], MambaSpec]: ...
@dataclasses.dataclass
class MambaCopyBuffers:
    src_ptrs: CpuGpuBuffer
    dst_ptrs: CpuGpuBuffer
    sizes: CpuGpuBuffer
    offset: int = ...
    @classmethod
    def create(
        cls,
        max_num_reqs: int,
        kv_cache_config: KVCacheConfig,
        copy_funcs: tuple[MambaStateCopyFunc, ...],
        make_buffer: Callable[..., CpuGpuBuffer],
    ) -> MambaCopyBuffers: ...

def collect_mamba_copy_meta(
    copy_bufs: MambaCopyBuffers,
    kv_cache_config: KVCacheConfig,
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    mamba_group_ids: list[int],
    src_block_idx: int,
    dest_block_idx: int,
    accept_token_bias: int,
    req_state: CachedRequestState,
    forward_context: dict[str, Any],
) -> None: ...
def do_mamba_copy_block(copy_bufs: MambaCopyBuffers): ...
def preprocess_mamba(
    scheduler_output: SchedulerOutput,
    kv_cache_config: KVCacheConfig,
    cache_config: CacheConfig,
    mamba_state_idx: dict[str, int],
    input_batch: GPUInputBatch,
    requests: dict[str, CachedRequestState],
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    copy_bufs: MambaCopyBuffers,
): ...
def postprocess_mamba(
    scheduler_output: SchedulerOutput,
    kv_cache_config: KVCacheConfig,
    input_batch: GPUInputBatch,
    requests: dict[str, CachedRequestState],
    mamba_state_idx: dict[str, int],
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    copy_bufs: MambaCopyBuffers,
): ...
