import dataclasses
import torch
from _typeshed import Incomplete
from collections.abc import Callable as Callable
from typing import Any
from vllm.compilation.counter import compilation_counter as compilation_counter
from vllm.compilation.monitor import (
    validate_cudagraph_capturing_enabled as validate_cudagraph_capturing_enabled,
)
from vllm.config import CUDAGraphMode as CUDAGraphMode, VllmConfig as VllmConfig
from vllm.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id as set_graph_pool_id,
)
from vllm.forward_context import (
    BatchDescriptor as BatchDescriptor,
    get_forward_context as get_forward_context,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.offloader.base import get_offloader as get_offloader
from vllm.platforms import current_platform as current_platform
from vllm.utils.torch_utils import (
    current_stream as current_stream,
    weak_ref_tensors as weak_ref_tensors,
)

logger: Incomplete

@dataclasses.dataclass(frozen=True)
class CUDAGraphStat:
    num_unpadded_tokens: int
    num_padded_tokens: int
    num_paddings: int
    runtime_mode: str

class CUDAGraphLogging:
    COLUMN_HEADERS: Incomplete
    cg_mode: Incomplete
    cg_capture_sizes: Incomplete
    settings_header: Incomplete
    def __init__(
        self, cg_mode: CUDAGraphMode, cg_capture_sizes: list[int] | None
    ) -> None: ...
    stats: list[CUDAGraphStat]
    def reset(self) -> None: ...
    def observe(self, cudagraph_stat: CUDAGraphStat) -> None: ...
    def generate_metric_table(self) -> str: ...
    def log(self, log_fn: Callable[..., Any] = ...) -> None: ...

@dataclasses.dataclass
class CUDAGraphEntry:
    batch_descriptor: BatchDescriptor
    cudagraph: torch.cuda.CUDAGraph | None = ...
    output: Any | None = ...
    input_addresses: list[int] | None = ...

@dataclasses.dataclass
class CUDAGraphOptions:
    debug_log_enable: bool = ...
    gc_disable: bool = ...
    weak_ref_output: bool = ...

class CUDAGraphWrapper:
    @classmethod
    def clear_all_graphs(cls) -> None: ...
    runnable: Incomplete
    vllm_config: Incomplete
    runtime_mode: Incomplete
    compilation_config: Incomplete
    first_run_finished: bool
    is_debugging_mode: Incomplete
    graph_pool: Incomplete
    cudagraph_options: Incomplete
    concrete_cudagraph_entries: dict[BatchDescriptor, CUDAGraphEntry]
    def __init__(
        self,
        runnable: Callable[..., Any],
        vllm_config: VllmConfig,
        runtime_mode: CUDAGraphMode,
        cudagraph_options: CUDAGraphOptions | None = None,
    ) -> None: ...
    def __getattr__(self, key: str) -> Any: ...
    def unwrap(self) -> Callable[..., Any]: ...
    @property
    def cudagraph_wrapper(self) -> CUDAGraphWrapper: ...
    def clear_graphs(self) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any | None: ...
