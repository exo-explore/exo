import torch
import types
from _typeshed import Incomplete
from typing import Any
from vllm import envs as envs
from vllm.distributed.device_communicators.pynccl import (
    PyNcclCommunicator as PyNcclCommunicator,
)
from vllm.logger import init_logger as init_logger
from vllm.platforms import current_platform as current_platform
from vllm.utils.nccl import find_nccl_include_paths as find_nccl_include_paths

logger: Incomplete
nccl_allocator_source: str

def is_symmetric_memory_enabled(): ...
def is_symmetric_memory_tensor(tensor: torch.Tensor): ...
def set_graph_pool_id(graph_pool_id: Any) -> None: ...
def compile_nccl_allocator() -> None: ...
def get_nccl_mem_pool(): ...

class nccl_symm_mem_context:
    disabled: Incomplete
    pynccl_comm: PyNcclCommunicator | None
    is_graph_capture: Incomplete
    device: Incomplete
    def __init__(
        self, pynccl_comm: PyNcclCommunicator, disabled: bool = False
    ) -> None: ...
    def __enter__(self): ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None: ...
