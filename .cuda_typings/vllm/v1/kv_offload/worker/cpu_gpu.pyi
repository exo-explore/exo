import numpy as np
import torch
from _typeshed import Incomplete
from dataclasses import dataclass
from vllm.logger import init_logger as init_logger
from vllm.utils.platform_utils import is_pin_memory_available as is_pin_memory_available
from vllm.v1.attention.backend import AttentionBackend as AttentionBackend
from vllm.v1.kv_offload.mediums import BlockIDsLoadStoreSpec as BlockIDsLoadStoreSpec
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler as OffloadingHandler,
    TransferResult as TransferResult,
    TransferSpec as TransferSpec,
)

logger: Incomplete

@dataclass
class Transfer:
    job_id: int
    stream: torch.cuda.Stream
    start_event: torch.Event
    end_event: torch.Event
    num_bytes: int

def expand_block_ids(
    block_ids: np.ndarray,
    block_size_factor: int,
    output: np.ndarray,
    skip_count: int = 0,
): ...

class SingleDirectionOffloadingHandler(OffloadingHandler):
    src_tensors: list[torch.Tensor]
    dst_tensors: list[torch.Tensor]
    src_block_size_factor: int
    dst_block_size_factor: int
    block_size_in_bytes: Incomplete
    total_block_size_in_bytes: Incomplete
    gpu_to_cpu: bool
    transfer_type: Incomplete
    def __init__(
        self,
        src_tensors: list[torch.Tensor],
        dst_tensors: list[torch.Tensor],
        src_block_size_factor: int,
        dst_block_size_factor: int,
    ) -> None: ...
    def transfer_async(self, job_id: int, transfer_spec: TransferSpec) -> bool: ...
    def get_finished(self) -> list[TransferResult]: ...
    def wait(self, job_ids: set[int]): ...

class CpuGpuOffloadingHandlers:
    gpu_to_cpu_handler: Incomplete
    cpu_to_gpu_handler: Incomplete
    def __init__(
        self,
        gpu_block_size: int,
        cpu_block_size: int,
        num_cpu_blocks: int,
        gpu_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ) -> None: ...
