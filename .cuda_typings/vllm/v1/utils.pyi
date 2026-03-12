import argparse
import numpy as np
import torch
from _typeshed import Incomplete
from collections.abc import Callable as Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from typing import Any, Generic, TypeVar, overload
from vllm.logger import init_logger as init_logger
from vllm.usage.usage_lib import (
    UsageContext as UsageContext,
    is_usage_stats_enabled as is_usage_stats_enabled,
    usage_message as usage_message,
)
from vllm.utils.network_utils import (
    get_open_port as get_open_port,
    get_open_zmq_ipc_path as get_open_zmq_ipc_path,
    get_tcp_uri as get_tcp_uri,
)
from vllm.utils.system_utils import kill_process_tree as kill_process_tree
from vllm.v1.core.sched.output import SchedulerOutput as SchedulerOutput
from vllm.v1.engine.coordinator import DPCoordinator as DPCoordinator
from vllm.v1.engine.utils import (
    CoreEngineActorManager as CoreEngineActorManager,
    CoreEngineProcManager as CoreEngineProcManager,
)

logger: Incomplete
T = TypeVar("T")

class ConstantList(Sequence, Generic[T]):
    def __init__(self, x: list[T]) -> None: ...
    def append(self, item) -> None: ...
    def extend(self, item) -> None: ...
    def insert(self, item) -> None: ...
    def pop(self, item) -> None: ...
    def remove(self, item) -> None: ...
    def clear(self) -> None: ...
    def index(self, item: T, start: int = 0, stop: int | None = None) -> int: ...
    @overload
    def __getitem__(self, item: int) -> T: ...
    @overload
    def __getitem__(self, s: slice) -> list[T]: ...
    @overload
    def __setitem__(self, item: int, value: T): ...
    @overload
    def __setitem__(self, s: slice, value: T): ...
    def __delitem__(self, item) -> None: ...
    def __iter__(self): ...
    def __contains__(self, item) -> bool: ...
    def __len__(self) -> int: ...
    def copy(self) -> list[T]: ...

class CpuGpuBuffer:
    cpu: Incomplete
    gpu: Incomplete
    np: np.ndarray
    def __init__(
        self,
        *size: int | torch.SymInt,
        dtype: torch.dtype,
        device: torch.device,
        pin_memory: bool,
        with_numpy: bool = True,
    ) -> None: ...
    def copy_to_gpu(self, n: int | None = None) -> torch.Tensor: ...
    def copy_to_cpu(self, n: int | None = None) -> torch.Tensor: ...

def get_engine_client_zmq_addr(local_only: bool, host: str, port: int = 0) -> str: ...

class APIServerProcessManager:
    listen_address: Incomplete
    sock: Incomplete
    args: Incomplete
    processes: list[BaseProcess]
    def __init__(
        self,
        target_server_fn: Callable,
        listen_address: str,
        sock: Any,
        args: argparse.Namespace,
        num_servers: int,
        input_addresses: list[str],
        output_addresses: list[str],
        stats_update_address: str | None = None,
    ) -> None: ...
    def close(self) -> None: ...

def wait_for_completion_or_failure(
    api_server_manager: APIServerProcessManager,
    engine_manager: CoreEngineProcManager | CoreEngineActorManager | None = None,
    coordinator: DPCoordinator | None = None,
) -> None: ...
def shutdown(procs: list[BaseProcess]): ...
def copy_slice(
    from_tensor: torch.Tensor, to_tensor: torch.Tensor, length: int
) -> torch.Tensor: ...
def report_usage_stats(vllm_config, usage_context: UsageContext = ...) -> None: ...
def record_function_or_nullcontext(name: str) -> AbstractContextManager: ...
def tensor_data(tensor: torch.Tensor) -> memoryview: ...
@dataclass
class IterationDetails:
    num_ctx_requests: int
    num_ctx_tokens: int
    num_generation_requests: int
    num_generation_tokens: int

def compute_iteration_details(
    scheduler_output: SchedulerOutput,
) -> IterationDetails: ...
