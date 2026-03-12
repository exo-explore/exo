import torch
import torch.nn as nn
from _typeshed import Incomplete
from collections.abc import Callable as Callable
from typing import Any
from vllm.config import (
    VllmConfig as VllmConfig,
    set_current_vllm_config as set_current_vllm_config,
)
from vllm.logger import init_logger as init_logger
from vllm.lora.request import LoRARequest as LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.tracing import instrument as instrument
from vllm.utils.import_utils import resolve_obj_by_qualname as resolve_obj_by_qualname
from vllm.utils.system_utils import (
    update_environment_variables as update_environment_variables,
)
from vllm.v1.core.sched.output import (
    GrammarOutput as GrammarOutput,
    SchedulerOutput as SchedulerOutput,
)
from vllm.v1.kv_cache_interface import KVCacheSpec as KVCacheSpec
from vllm.v1.outputs import (
    AsyncModelRunnerOutput as AsyncModelRunnerOutput,
    ModelRunnerOutput as ModelRunnerOutput,
)

logger: Incomplete

class WorkerBase:
    vllm_config: Incomplete
    model_config: Incomplete
    cache_config: Incomplete
    lora_config: Incomplete
    load_config: Incomplete
    parallel_config: Incomplete
    scheduler_config: Incomplete
    device_config: Incomplete
    speculative_config: Incomplete
    observability_config: Incomplete
    kv_transfer_config: Incomplete
    compilation_config: Incomplete
    current_platform: Incomplete
    local_rank: Incomplete
    rank: Incomplete
    distributed_init_method: Incomplete
    is_driver_worker: Incomplete
    device: torch.device | None
    model_runner: nn.Module | None
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None: ...
    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]: ...
    def compile_or_warm_up_model(self) -> float: ...
    def check_health(self) -> None: ...
    def init_device(self) -> None: ...
    def reset_mm_cache(self) -> None: ...
    def get_model(self) -> nn.Module: ...
    def apply_model(self, fn: Callable[[nn.Module], _R]) -> _R: ...
    def get_model_inspection(self) -> str: ...
    def load_model(self) -> None: ...
    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None: ...
    def sample_tokens(
        self, grammar_output: GrammarOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput: ...
    def get_cache_block_size_bytes(self) -> int: ...
    def add_lora(self, lora_request: LoRARequest) -> bool: ...
    def remove_lora(self, lora_id: int) -> bool: ...
    def pin_lora(self, lora_id: int) -> bool: ...
    def list_loras(self) -> set[int]: ...
    @property
    def vocab_size(self) -> int: ...
    def shutdown(self) -> None: ...

class WorkerWrapperBase:
    rpc_rank: Incomplete
    global_rank: Incomplete
    worker: WorkerBase
    vllm_config: VllmConfig
    def __init__(self, rpc_rank: int = 0, global_rank: int | None = None) -> None: ...
    def shutdown(self) -> None: ...
    def update_environment_variables(self, envs_list: list[dict[str, str]]) -> None: ...
    mm_receiver_cache: Incomplete
    def init_worker(self, all_kwargs: list[dict[str, Any]]) -> None: ...
    def initialize_from_config(self, kv_cache_configs: list[Any]) -> None: ...
    def init_device(self) -> None: ...
    def __getattr__(self, attr: str): ...
    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None: ...
    def reset_mm_cache(self) -> None: ...
