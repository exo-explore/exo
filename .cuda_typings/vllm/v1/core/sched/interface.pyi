import abc
import enum
from abc import ABC, abstractmethod
from collections.abc import Iterable
from vllm.config import VllmConfig as VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1 as KVConnectorBase_V1,
)
from vllm.multimodal import (
    MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY,
    MultiModalRegistry as MultiModalRegistry,
)
from vllm.v1.core.sched.output import (
    GrammarOutput as GrammarOutput,
    SchedulerOutput as SchedulerOutput,
)
from vllm.v1.engine import EngineCoreOutputs as EngineCoreOutputs
from vllm.v1.kv_cache_interface import KVCacheConfig as KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats as SchedulerStats
from vllm.v1.outputs import (
    DraftTokenIds as DraftTokenIds,
    ModelRunnerOutput as ModelRunnerOutput,
)
from vllm.v1.request import Request as Request, RequestStatus as RequestStatus
from vllm.v1.structured_output import StructuredOutputManager as StructuredOutputManager

class PauseState(enum.IntEnum):
    UNPAUSED = 0
    PAUSED_NEW = 1
    PAUSED_ALL = 2

class SchedulerInterface(ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        mm_registry: MultiModalRegistry = ...,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ): ...
    @abstractmethod
    def schedule(self) -> SchedulerOutput: ...
    @abstractmethod
    def get_grammar_bitmask(
        self, scheduler_output: SchedulerOutput
    ) -> GrammarOutput | None: ...
    @abstractmethod
    def update_from_output(
        self, scheduler_output: SchedulerOutput, model_runner_output: ModelRunnerOutput
    ) -> dict[int, "EngineCoreOutputs"]: ...
    @abstractmethod
    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None: ...
    @abstractmethod
    def update_draft_token_ids_in_output(
        self, draft_token_ids: DraftTokenIds, scheduler_output: SchedulerOutput
    ) -> None: ...
    @abstractmethod
    def add_request(self, request: Request) -> None: ...
    @abstractmethod
    def finish_requests(
        self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus
    ) -> list[tuple[str, int]]: ...
    @abstractmethod
    def get_num_unfinished_requests(self) -> int: ...
    def has_unfinished_requests(self) -> bool: ...
    @abstractmethod
    def has_finished_requests(self) -> bool: ...
    def has_requests(self) -> bool: ...
    @property
    @abstractmethod
    def pause_state(self) -> PauseState: ...
    @abstractmethod
    def set_pause_state(self, pause_state: PauseState) -> None: ...
    @abstractmethod
    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool: ...
    @abstractmethod
    def reset_encoder_cache(self) -> None: ...
    @abstractmethod
    def get_request_counts(self) -> tuple[int, int]: ...
    @abstractmethod
    def make_stats(self) -> SchedulerStats | None: ...
    @abstractmethod
    def shutdown(self) -> None: ...
    def get_kv_connector(self) -> KVConnectorBase_V1 | None: ...
