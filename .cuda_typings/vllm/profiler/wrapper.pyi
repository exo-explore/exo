import abc
import torch
from _typeshed import Incomplete
from abc import ABC
from collections.abc import Callable as Callable
from typing_extensions import override
from vllm.config import ProfilerConfig as ProfilerConfig
from vllm.logger import init_logger as init_logger

logger: Incomplete

class WorkerProfiler(ABC, metaclass=abc.ABCMeta):
    def __init__(self, profiler_config: ProfilerConfig) -> None: ...
    def start(self) -> None: ...
    def step(self) -> None: ...
    def stop(self) -> None: ...
    def shutdown(self) -> None: ...
    def annotate_context_manager(self, name: str): ...

TorchProfilerActivity: Incomplete
TorchProfilerActivityMap: Incomplete

class TorchProfilerWrapper(WorkerProfiler):
    local_rank: Incomplete
    profiler_config: Incomplete
    dump_cpu_time_total: Incomplete
    profiler: Incomplete
    def __init__(
        self,
        profiler_config: ProfilerConfig,
        worker_name: str,
        local_rank: int,
        activities: list[TorchProfilerActivity],
        on_trace_ready: Callable[[torch.profiler.profile], None] | None = None,
    ) -> None: ...
    @override
    def annotate_context_manager(self, name: str): ...

class CudaProfilerWrapper(WorkerProfiler):
    def __init__(self, profiler_config: ProfilerConfig) -> None: ...
    @override
    def annotate_context_manager(self, name: str): ...
