import enum
import torch
import zmq
from _typeshed import Incomplete
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal
from vllm.config import VllmConfig as VllmConfig
from vllm.distributed.kv_events import KVCacheEvent as KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1 as KVConnectorBase_V1,
    KVConnectorMetadata as KVConnectorMetadata,
    KVConnectorRole as KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration import (
    LMCacheMPSchedulerAdapter as LMCacheMPSchedulerAdapter,
    LMCacheMPWorkerAdapter as LMCacheMPWorkerAdapter,
    LoadStoreOp as LoadStoreOp,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics as KVConnectorPromMetrics,
    KVConnectorStats as KVConnectorStats,
    PromMetric as PromMetric,
    PromMetricT as PromMetricT,
)
from vllm.forward_context import ForwardContext as ForwardContext
from vllm.v1.attention.backend import AttentionMetadata as AttentionMetadata
from vllm.v1.core.kv_cache_manager import KVCacheBlocks as KVCacheBlocks
from vllm.v1.core.kv_cache_utils import BlockHash as BlockHash
from vllm.v1.core.sched.output import SchedulerOutput as SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig as KVCacheConfig
from vllm.v1.outputs import KVConnectorOutput as KVConnectorOutput
from vllm.v1.request import Request as Request, RequestStatus as RequestStatus
from vllm.v1.utils import ConstantList as ConstantList

logger: Incomplete

def reformat_block_ids(block_ids: tuple[list[int], ...] | None) -> list[int]: ...
def extract_world_size_and_kv_rank(
    world_size: int, rank: int, vllm_config: VllmConfig
) -> tuple[int, int]: ...
def create_scheduler_adapter(
    server_url: str, zmq_context: zmq.Context, vllm_config: VllmConfig
) -> LMCacheMPSchedulerAdapter: ...
def create_worker_adapter(
    server_url: str, zmq_context: zmq.Context, vllm_config: VllmConfig
) -> LMCacheMPWorkerAdapter: ...

class LMCacheMPRequestState(enum.Enum):
    PREFETCHING = ...
    WAITING_FOR_LOAD = ...
    READY = ...

@dataclass
class LMCacheMPRequestTracker:
    request_id: str
    all_token_ids: ConstantList[int]
    block_hashes: ConstantList["BlockHash"]
    allocated_block_ids: list[int] = field(default_factory=list)
    num_scheduled_tokens: int = ...
    num_stored_blocks: int = ...
    num_vllm_hit_blocks: int = ...
    num_lmcache_hit_blocks: int = ...
    state: LMCacheMPRequestState = ...
    def __init__(self, request: Request) -> None: ...
    def needs_retrieve(self) -> bool: ...
    def is_ready_for_retrieving(self) -> bool: ...
    def increase_num_scheduled_tokens(self, num_new_tokens: int): ...
    def increase_num_stored_blocks(self, num_new_blocks: int): ...
    def append_block_ids(self, new_block_ids: list[int]): ...

@dataclass
class LMCacheMPRequestMetadata:
    request_id: str
    direction: Literal["STORE", "RETRIEVE"]
    op: LoadStoreOp
    @staticmethod
    def GetStoreMetadata(
        tracker: LMCacheMPRequestTracker, blocks_in_chunk: int, vllm_block_size: int
    ) -> LMCacheMPRequestMetadata | None: ...
    @staticmethod
    def GetRetrieveMetadata(
        tracker: LMCacheMPRequestTracker, blocks_in_chunk: int, vllm_block_size: int
    ) -> LMCacheMPRequestMetadata | None: ...

class LMCacheMPConnectorMetadata(KVConnectorMetadata):
    requests: list[LMCacheMPRequestMetadata]
    def __init__(self) -> None: ...
    def add_request_metadata(self, request_metadata: LMCacheMPRequestMetadata): ...
    def __len__(self) -> int: ...

class LMCacheMPConnector(KVConnectorBase_V1):
    scheduler_adapter: Incomplete
    request_trackers: dict[str, LMCacheMPRequestTracker]
    worker_adapter: Incomplete
    vllm_block_size: Incomplete
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ) -> None: ...
    @property
    def role(self) -> KVConnectorRole: ...
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]): ...
    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None: ...
    def wait_for_layer_load(self, layer_name: str) -> None: ...
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None: ...
    def wait_for_save(self) -> None: ...
    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]: ...
    def get_block_ids_with_load_errors(self) -> set[int]: ...
    def shutdown(self) -> None: ...
    def get_kv_connector_stats(self) -> KVConnectorStats | None: ...
    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]: ...
    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ): ...
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata: ...
    def update_connector_output(self, connector_output: KVConnectorOutput): ...
    def request_finished(
        self, request: Request, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]: ...
    def take_events(self) -> Iterable["KVCacheEvent"]: ...
    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig) -> str | None: ...
    def get_finished_count(self) -> int | None: ...
    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> KVConnectorStats | None: ...
    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: VllmConfig,
        metric_types: dict[type["PromMetric"], type["PromMetricT"]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> KVConnectorPromMetrics | None: ...
