from _typeshed import Incomplete
from collections.abc import Callable as Callable
from typing import Any
from vllm import envs as envs
from vllm.config import VllmConfig as VllmConfig
from vllm.logger import init_logger as init_logger
from vllm.platforms import (
    CpuArchEnum as CpuArchEnum,
    current_platform as current_platform,
)
from vllm.platforms.cpu import (
    CpuPlatform as CpuPlatform,
    LogicalCPUInfo as LogicalCPUInfo,
)
from vllm.profiler.wrapper import TorchProfilerWrapper as TorchProfilerWrapper
from vllm.utils.torch_utils import set_random_seed as set_random_seed
from vllm.v1.worker.cpu_model_runner import CPUModelRunner as CPUModelRunner
from vllm.v1.worker.gpu_worker import (
    Worker as Worker,
    init_worker_distributed_environment as init_worker_distributed_environment,
)

logger: Incomplete

class CPUWorker(Worker):
    profiler: Any | None
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None: ...
    local_omp_cpuid: Incomplete
    model_runner: CPUModelRunner
    def init_device(self): ...
    def sleep(self, level: int = 1) -> None: ...
    def wake_up(self, tags: list[str] | None = None) -> None: ...
    def determine_available_memory(self) -> int: ...
    def compile_or_warm_up_model(self) -> float: ...
    def profile(self, is_start: bool = True, profile_prefix: str | None = None): ...
