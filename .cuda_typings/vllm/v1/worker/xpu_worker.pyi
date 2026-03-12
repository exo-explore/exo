from .utils import request_memory as request_memory
from _typeshed import Incomplete
from typing import Any
from vllm.config import VllmConfig as VllmConfig
from vllm.logger import init_logger as init_logger
from vllm.platforms import current_platform as current_platform
from vllm.profiler.wrapper import TorchProfilerWrapper as TorchProfilerWrapper
from vllm.utils.mem_utils import (
    MemorySnapshot as MemorySnapshot,
    format_gib as format_gib,
)
from vllm.utils.torch_utils import set_random_seed as set_random_seed
from vllm.v1.utils import report_usage_stats as report_usage_stats
from vllm.v1.worker.gpu_worker import (
    Worker as Worker,
    init_worker_distributed_environment as init_worker_distributed_environment,
)
from vllm.v1.worker.workspace import init_workspace_manager as init_workspace_manager
from vllm.v1.worker.xpu_model_runner import (
    XPUModelRunner as XPUModelRunner,
    XPUModelRunnerV2 as XPUModelRunnerV2,
)

logger: Incomplete

class XPUWorker(Worker):
    profiler: Any | None
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None: ...
    device: Incomplete
    init_gpu_memory: Incomplete
    init_snapshot: Incomplete
    requested_memory: Incomplete
    model_runner: Incomplete
    def init_device(self) -> None: ...
