import abc
import torch
import torch.nn as nn
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from vllm.logger import init_logger as init_logger
from vllm.model_executor.offloader.base import BaseOffloader as BaseOffloader
from vllm.utils.platform_utils import is_pin_memory_available as is_pin_memory_available

logger: Incomplete

@dataclass
class ParamInfo:
    name: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    @property
    def key(self) -> tuple[str, tuple[int, ...], tuple[int, ...], torch.dtype]: ...
    @property
    def num_bytes(self) -> int: ...

class StaticBufferPool:
    slot_capacity: Incomplete
    total_bytes: int
    def __init__(
        self, param_infos: list[ParamInfo], slot_capacity: int, device: torch.device
    ) -> None: ...
    def get_buffer(
        self,
        name: str,
        shape: tuple[int, ...],
        stride: tuple[int, ...],
        dtype: torch.dtype,
        slot_idx: int,
    ) -> torch.Tensor: ...

class PrefetchOffloader(BaseOffloader):
    group_size: Incomplete
    num_in_group: Incomplete
    prefetch_step: Incomplete
    offload_params: Incomplete
    mode: Incomplete
    copy_stream: Incomplete
    module_offloaders: list[_ModuleOffloader]
    buffer_pool: StaticBufferPool | None
    total_offloaded_bytes: int
    def __init__(
        self,
        group_size: int,
        num_in_group: int,
        prefetch_step: int,
        offload_params: set[str] | None = None,
        mode: str = "cpu",
    ) -> None: ...
    def wrap_modules(
        self, modules_generator: Generator[nn.Module, None, None]
    ) -> list[nn.Module]: ...
    def sync_prev_onload(self) -> None: ...
    def join_after_forward(self) -> None: ...
    def post_init(self) -> None: ...

class _ModuleOffloader:
    mode: Incomplete
    module: Incomplete
    device: Incomplete
    copy_stream: Incomplete
    layer_idx: Incomplete
    offloaded_bytes: int
    def __init__(
        self,
        mode: str,
        module: nn.Module,
        copy_stream: torch.cuda.Stream,
        whitelist_param_names: list[str],
        layer_idx: int,
    ) -> None: ...
    def post_init(self) -> None: ...
    def sync_cpu_storage(self) -> None: ...
    def get_param_infos(self) -> list[ParamInfo]: ...
    def assign_buffer_slot(self, pool: StaticBufferPool, slot_idx: int): ...
    def start_onload_to_static(self) -> None: ...

class _BaseParamOffloader(ABC, metaclass=abc.ABCMeta):
    @staticmethod
    def create(mode: str, **kwargs) -> _BaseParamOffloader: ...
    offloaded_bytes: int
    def __init__(self, module: nn.Module, param_name: str) -> None: ...
    def post_init(self) -> None: ...
    @abstractmethod
    def sync_cpu_storage(self) -> None: ...
    @abstractmethod
    def assign_static_buffer(self, gpu_buffer: torch.Tensor) -> None: ...

class _CpuParamOffloader(_BaseParamOffloader):
    def __init__(self, module: nn.Module, param_name: str) -> None: ...
    def assign_static_buffer(self, gpu_buffer: torch.Tensor) -> None: ...
    def sync_cpu_storage(self) -> None: ...
    def post_init(self) -> None: ...
