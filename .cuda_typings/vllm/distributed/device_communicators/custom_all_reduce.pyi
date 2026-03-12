import torch
from _typeshed import Incomplete
from collections.abc import Generator
from contextlib import contextmanager
from torch.distributed import ProcessGroup as ProcessGroup
from vllm.distributed.device_communicators.all_reduce_utils import (
    CUSTOM_ALL_REDUCE_MAX_SIZES as CUSTOM_ALL_REDUCE_MAX_SIZES,
    gpu_p2p_access_check as gpu_p2p_access_check,
)
from vllm.distributed.parallel_state import in_the_same_node_as as in_the_same_node_as
from vllm.logger import init_logger as init_logger
from vllm.platforms import current_platform as current_platform
from vllm.utils.torch_utils import (
    cuda_device_count_stateless as cuda_device_count_stateless,
)

custom_ar: bool
logger: Incomplete

def is_weak_contiguous(inp: torch.Tensor): ...

class CustomAllreduce:
    disabled: bool
    group: Incomplete
    rank: Incomplete
    device: Incomplete
    meta_ptrs: Incomplete
    buffer_ptrs: Incomplete
    rank_data: Incomplete
    max_size: Incomplete
    world_size: Incomplete
    fully_connected: Incomplete
    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        max_size=...,
        symm_mem_enabled: bool = False,
    ) -> None: ...
    @contextmanager
    def capture(self) -> Generator[None]: ...
    def register_graph_buffers(self) -> None: ...
    def should_custom_ar(self, inp: torch.Tensor): ...
    def all_reduce(
        self, inp: torch.Tensor, *, out: torch.Tensor = None, registered: bool = False
    ): ...
    def custom_all_reduce(self, input: torch.Tensor) -> torch.Tensor | None: ...
    def close(self) -> None: ...
    def __del__(self) -> None: ...
    @staticmethod
    def create_shared_buffer(
        size_in_bytes: int,
        group: ProcessGroup | None = None,
        uncached: bool | None = False,
    ) -> list[int]: ...
    @staticmethod
    def free_shared_buffer(
        pointers: list[int], group: ProcessGroup | None = None, rank: int | None = None
    ) -> None: ...
