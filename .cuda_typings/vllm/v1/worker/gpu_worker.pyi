import torch
import torch.nn as nn
from ...model_executor.model_loader import TensorizerLoader as TensorizerLoader
from .gpu.warmup import warmup_kernels as warmup_kernels
from .utils import request_memory as request_memory
from _typeshed import Incomplete
from collections.abc import Callable as Callable
from typing import Any
from vllm.config import (
    CUDAGraphMode as CUDAGraphMode,
    VllmConfig as VllmConfig,
    set_current_vllm_config as set_current_vllm_config,
)
from vllm.config.compilation import CompilationMode as CompilationMode
from vllm.distributed import (
    ensure_model_parallel_initialized as ensure_model_parallel_initialized,
    init_distributed_environment as init_distributed_environment,
    set_custom_all_reduce as set_custom_all_reduce,
)
from vllm.distributed.ec_transfer import (
    ensure_ec_transfer_initialized as ensure_ec_transfer_initialized,
)
from vllm.distributed.eplb.eplb_utils import (
    override_envs_for_eplb as override_envs_for_eplb,
)
from vllm.distributed.kv_transfer import (
    ensure_kv_transfer_initialized as ensure_kv_transfer_initialized,
    ensure_kv_transfer_shutdown as ensure_kv_transfer_shutdown,
    get_kv_transfer_group as get_kv_transfer_group,
    has_kv_transfer_group as has_kv_transfer_group,
)
from vllm.distributed.parallel_state import (
    Handle as Handle,
    get_pp_group as get_pp_group,
    get_tp_group as get_tp_group,
)
from vllm.distributed.weight_transfer import (
    WeightTransferEngineFactory as WeightTransferEngineFactory,
)
from vllm.logger import init_logger as init_logger
from vllm.lora.request import LoRARequest as LoRARequest
from vllm.model_executor.model_loader.tensorizer import (
    TensorizerConfig as TensorizerConfig,
)
from vllm.model_executor.warmup.kernel_warmup import kernel_warmup as kernel_warmup
from vllm.platforms import current_platform as current_platform
from vllm.profiler.wrapper import (
    CudaProfilerWrapper as CudaProfilerWrapper,
    TorchProfilerWrapper as TorchProfilerWrapper,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors
from vllm.tasks import SupportedTask as SupportedTask
from vllm.tracing import instrument as instrument
from vllm.utils.mem_constants import GiB_bytes as GiB_bytes
from vllm.utils.mem_utils import (
    MemorySnapshot as MemorySnapshot,
    format_gib as format_gib,
    memory_profiling as memory_profiling,
)
from vllm.utils.torch_utils import set_random_seed as set_random_seed
from vllm.v1.core.sched.output import (
    GrammarOutput as GrammarOutput,
    SchedulerOutput as SchedulerOutput,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig as KVCacheConfig,
    KVCacheSpec as KVCacheSpec,
)
from vllm.v1.outputs import (
    AsyncModelRunnerOutput as AsyncModelRunnerOutput,
    DraftTokenIds as DraftTokenIds,
    ModelRunnerOutput as ModelRunnerOutput,
)
from vllm.v1.utils import (
    compute_iteration_details as compute_iteration_details,
    report_usage_stats as report_usage_stats,
)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GPUModelRunner
from vllm.v1.worker.utils import (
    is_residual_scattered_for_sp as is_residual_scattered_for_sp,
)
from vllm.v1.worker.worker_base import WorkerBase as WorkerBase
from vllm.v1.worker.workspace import init_workspace_manager as init_workspace_manager

logger: Incomplete

class AsyncIntermediateTensors(IntermediateTensors):
    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
        comm_handles: list[Handle] | None = None,
        comm_postprocess: list[Callable[[], None]] | None = None,
    ) -> None: ...
    def wait_for_comm(self) -> None: ...
    def __getattribute__(self, name: str): ...

class Worker(WorkerBase):
    elastic_ep_executor: Incomplete
    weight_transfer_engine: Incomplete
    profiler: Any | None
    profiler_config: Incomplete
    use_v2_model_runner: Incomplete
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None: ...
    def sleep(self, level: int = 1) -> None: ...
    def wake_up(self, tags: list[str] | None = None) -> None: ...
    device: Incomplete
    init_snapshot: Incomplete
    requested_memory: Incomplete
    model_runner: GPUModelRunner
    def init_device(self) -> None: ...
    def load_model(self) -> None: ...
    def update_config(self, overrides: dict[str, Any]) -> None: ...
    def reload_weights(self, *args, **kwargs) -> None: ...
    non_torch_memory: Incomplete
    peak_activation_memory: Incomplete
    cudagraph_memory_estimate: Incomplete
    available_kv_cache_memory_bytes: Incomplete
    def determine_available_memory(self) -> int: ...
    def get_kv_connector_handshake_metadata(self) -> dict | None: ...
    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]: ...
    def update_max_model_len(self, max_model_len: int) -> None: ...
    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None: ...
    def compile_or_warm_up_model(self) -> float: ...
    def reset_mm_cache(self) -> None: ...
    def reset_encoder_cache(self) -> None: ...
    def get_model(self) -> nn.Module: ...
    def get_supported_tasks(self) -> tuple[SupportedTask, ...]: ...
    def get_encoder_timing_stats(self) -> dict[str, dict[str, float | int]]: ...
    def annotate_profile(self, scheduler_output): ...
    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput: ...
    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None: ...
    def take_draft_token_ids(self) -> DraftTokenIds | None: ...
    def profile(self, is_start: bool = True, profile_prefix: str | None = None): ...
    def execute_dummy_batch(self) -> None: ...
    def add_lora(self, lora_request: LoRARequest) -> bool: ...
    def remove_lora(self, lora_id: int) -> bool: ...
    def list_loras(self) -> set[int]: ...
    def pin_lora(self, lora_id: int) -> bool: ...
    def check_health(self) -> None: ...
    def save_sharded_state(
        self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None: ...
    def save_tensorized_model(self, tensorizer_config: TensorizerConfig) -> None: ...
    def init_weight_transfer_engine(self, init_info: dict) -> None: ...
    def update_weights(self, update_info: dict) -> None: ...
    def shutdown(self) -> None: ...
    def elastic_ep_execute(self, execute_method: str, *args, **kwargs): ...

def init_worker_distributed_environment(
    vllm_config: VllmConfig,
    rank: int,
    distributed_init_method: str | None = None,
    local_rank: int = -1,
    backend: str = "nccl",
) -> None: ...
