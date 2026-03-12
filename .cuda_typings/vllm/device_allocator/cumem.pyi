import dataclasses
import torch
from _typeshed import Incomplete
from collections.abc import Callable as Callable, Iterator
from contextlib import contextmanager
from typing import Any
from vllm.cumem_allocator import (
    init_module as init_module,
    python_create_and_map as python_create_and_map,
    python_unmap_and_release as python_unmap_and_release,
)
from vllm.distributed.device_communicators.cuda_wrapper import (
    CudaRTLibrary as CudaRTLibrary,
)
from vllm.logger import init_logger as init_logger
from vllm.utils.platform_utils import is_pin_memory_available as is_pin_memory_available
from vllm.utils.system_utils import find_loaded_library as find_loaded_library

logger: Incomplete
cumem_available: bool
libcudart: Any
lib_name: Incomplete
HandleType = tuple[int, int, int, int]

@dataclasses.dataclass
class AllocationData:
    handle: HandleType
    tag: str
    cpu_backup_tensor: torch.Tensor | None = ...

def create_and_map(allocation_handle: HandleType) -> None: ...
def unmap_and_release(allocation_handle: HandleType) -> None: ...
def get_pluggable_allocator(
    python_malloc_fn: Callable[[HandleType], None],
    python_free_func: Callable[[int], HandleType],
) -> torch.cuda.memory.CUDAPluggableAllocator: ...
@contextmanager
def use_memory_pool_with_allocator(
    python_malloc_fn: Callable[[HandleType], None],
    python_free_func: Callable[[int], HandleType],
) -> Iterator[
    tuple[torch.cuda.memory.MemPool, torch.cuda.memory.CUDAPluggableAllocator]
]: ...

class CuMemAllocator:
    instance: CuMemAllocator | None
    default_tag: str
    @staticmethod
    def get_instance() -> CuMemAllocator: ...
    pointer_to_data: dict[int, AllocationData]
    current_tag: str
    allocator_and_pools: dict[str, Any]
    python_malloc_callback: Incomplete
    python_free_callback: Incomplete
    def __init__(self) -> None: ...
    def sleep(self, offload_tags: tuple[str, ...] | str | None = None) -> None: ...
    def wake_up(self, tags: list[str] | None = None) -> None: ...
    @contextmanager
    def use_memory_pool(self, tag: str | None = None): ...
    def get_current_usage(self) -> int: ...
