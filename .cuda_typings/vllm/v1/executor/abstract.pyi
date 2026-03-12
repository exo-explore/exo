import abc
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import Future
from functools import cached_property as cached_property
from typing import Literal, overload
from vllm.config import VllmConfig as VllmConfig
from vllm.distributed.kv_transfer.kv_connector.base import (
    KVConnectorBase as KVConnectorBase,
)
from vllm.distributed.kv_transfer.kv_connector.utils import (
    KVOutputAggregator as KVOutputAggregator,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata as KVConnectorHandshakeMetadata,
)
from vllm.logger import init_logger as init_logger
from vllm.lora.request import LoRARequest as LoRARequest
from vllm.tasks import SupportedTask as SupportedTask
from vllm.tracing import instrument as instrument
from vllm.utils.import_utils import resolve_obj_by_qualname as resolve_obj_by_qualname
from vllm.v1.core.sched.output import (
    GrammarOutput as GrammarOutput,
    SchedulerOutput as SchedulerOutput,
)
from vllm.v1.engine import (
    ReconfigureDistributedRequest as ReconfigureDistributedRequest,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig as KVCacheConfig,
    KVCacheSpec as KVCacheSpec,
)
from vllm.v1.outputs import (
    DraftTokenIds as DraftTokenIds,
    ModelRunnerOutput as ModelRunnerOutput,
)
from vllm.v1.worker.worker_base import WorkerBase as WorkerBase

logger: Incomplete
FailureCallback = Callable[[], None]

class Executor(ABC, metaclass=abc.ABCMeta):
    uses_ray: bool
    supports_pp: bool
    @staticmethod
    def get_class(vllm_config: VllmConfig) -> type["Executor"]: ...
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
    is_sleeping: bool
    sleeping_tags: set[str]
    kv_output_aggregator: KVOutputAggregator | None
    def __init__(self, vllm_config: VllmConfig) -> None: ...
    def initialize_from_config(self, kv_cache_configs: list[KVCacheConfig]) -> None: ...
    def register_failure_callback(self, callback: FailureCallback): ...
    def determine_available_memory(self) -> list[int]: ...
    def get_kv_cache_specs(self) -> list[dict[str, KVCacheSpec]]: ...
    @overload
    def collective_rpc(
        self,
        method: str | Callable[[WorkerBase], _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: Literal[False] = False,
    ) -> list[_R]: ...
    @overload
    def collective_rpc(
        self,
        method: str | Callable[[WorkerBase], _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: Literal[True] = True,
    ) -> Future[list[_R]]: ...
    def get_kv_connector_handshake_metadata(
        self,
    ) -> list[dict[int, KVConnectorHandshakeMetadata]]: ...
    @overload
    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: Literal[False] = False
    ) -> ModelRunnerOutput | None: ...
    @overload
    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: Literal[True] = True
    ) -> Future[ModelRunnerOutput | None]: ...
    @overload
    def sample_tokens(
        self, grammar_output: GrammarOutput | None, non_block: Literal[False] = False
    ) -> ModelRunnerOutput: ...
    @overload
    def sample_tokens(
        self, grammar_output: GrammarOutput | None, non_block: Literal[True] = True
    ) -> Future[ModelRunnerOutput]: ...
    def execute_dummy_batch(self) -> None: ...
    def take_draft_token_ids(self) -> DraftTokenIds | None: ...
    @property
    def max_concurrent_batches(self) -> int: ...
    def profile(self, is_start: bool = True, profile_prefix: str | None = None): ...
    def save_sharded_state(
        self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None: ...
    @abstractmethod
    def check_health(self) -> None: ...
    def shutdown(self) -> None: ...
    def init_kv_output_aggregator(self, connector: KVConnectorBase) -> None: ...
    @cached_property
    def supported_tasks(self) -> tuple[SupportedTask, ...]: ...
    def add_lora(self, lora_request: LoRARequest) -> bool: ...
    def remove_lora(self, lora_id: int) -> bool: ...
    def pin_lora(self, lora_id: int) -> bool: ...
    def list_loras(self) -> set[int]: ...
    def reset_mm_cache(self) -> None: ...
    def reset_encoder_cache(self) -> None: ...
    def sleep(self, level: int = 1): ...
    def wake_up(self, tags: list[str] | None = None): ...
    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None: ...

UniProcExecutor: Incomplete
ExecutorWithExternalLauncher: Incomplete
