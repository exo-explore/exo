import torch
from _typeshed import Incomplete
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from vllm.config import (
    VllmConfig as VllmConfig,
    get_layers_from_vllm_config as get_layers_from_vllm_config,
)
from vllm.distributed.kv_events import (
    BlockRemoved as BlockRemoved,
    BlockStored as BlockStored,
    KVCacheEvent as KVCacheEvent,
)
from vllm.distributed.kv_transfer.kv_connector.utils import (
    yield_req_data as yield_req_data,
)
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1 as KVConnectorBase_V1,
    KVConnectorRole as KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata as KVConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics as KVConnectorPromMetrics,
    KVConnectorStats as KVConnectorStats,
    PromMetric as PromMetric,
    PromMetricT as PromMetricT,
)
from vllm.forward_context import ForwardContext as ForwardContext
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.v1.attention.backend import (
    AttentionBackend as AttentionBackend,
    AttentionMetadata as AttentionMetadata,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks as KVCacheBlocks
from vllm.v1.core.kv_cache_utils import BlockHash as BlockHash
from vllm.v1.core.sched.output import SchedulerOutput as SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig as KVCacheConfig
from vllm.v1.kv_offload.abstract import OffloadingManager as OffloadingManager
from vllm.v1.kv_offload.factory import OffloadingSpecFactory as OffloadingSpecFactory
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec as GPULoadStoreSpec
from vllm.v1.kv_offload.spec import OffloadingSpec as OffloadingSpec
from vllm.v1.kv_offload.worker.worker import (
    OffloadingWorker as OffloadingWorker,
    TransferSpec as TransferSpec,
    TransferType as TransferType,
)
from vllm.v1.outputs import KVConnectorOutput as KVConnectorOutput
from vllm.v1.request import Request as Request

ReqId = str
logger: Incomplete

@dataclass
class OffloadingOperationMetrics:
    op_size: int
    op_time: float

@dataclass
class OffloadingConnectorStats(KVConnectorStats):
    def __post_init__(self) -> None: ...
    data: dict[str, list[OffloadingOperationMetrics]] = ...
    def reset(self) -> None: ...
    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats: ...
    def reduce(self) -> dict[str, int | float]: ...
    def is_empty(self) -> bool: ...
    def record_transfer(
        self, num_bytes: int, time: float, transfer_type: TransferType
    ): ...

@dataclass
class OffloadingConnectorMetadata(KVConnectorMetadata):
    reqs_to_load: dict[ReqId, TransferSpec]
    reqs_to_store: dict[ReqId, TransferSpec]

class OffloadingConnector(KVConnectorBase_V1):
    @property
    def prefer_cross_layer_blocks(self) -> bool: ...
    connector_scheduler: OffloadingConnectorScheduler | None
    connector_worker: OffloadingConnectorWorker | None
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ) -> None: ...
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]): ...
    def register_cross_layers_kv_cache(
        self, kv_cache: torch.Tensor, attn_backend: type[AttentionBackend]
    ): ...
    def handle_preemptions(self, preempted_req_ids: set[str]): ...
    def start_load_kv(self, forward_context: ForwardContext, **kwargs) -> None: ...
    def wait_for_layer_load(self, layer_name: str) -> None: ...
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> None: ...
    def wait_for_save(self) -> None: ...
    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]: ...
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
    def take_events(self) -> Iterable[KVCacheEvent]: ...
    def get_kv_connector_stats(self) -> KVConnectorStats | None: ...
    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> KVConnectorStats | None: ...
    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> KVConnectorPromMetrics: ...

class OffloadingConnectorScheduler:
    gpu_block_size: Incomplete
    offloaded_block_size: Incomplete
    block_size_factor: Incomplete
    manager: OffloadingManager
    def __init__(self, spec: OffloadingSpec) -> None: ...
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
    def take_events(self) -> Iterable[KVCacheEvent]: ...

class OffloadingConnectorWorker:
    spec: Incomplete
    worker: Incomplete
    kv_connector_stats: Incomplete
    def __init__(self, spec: OffloadingSpec) -> None: ...
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]): ...
    def register_cross_layers_kv_cache(
        self, kv_cache: torch.Tensor, attn_backend: type[AttentionBackend]
    ): ...
    def handle_preemptions(self, preempted_req_ids: set[str]): ...
    def start_kv_transfers(self, metadata: OffloadingConnectorMetadata): ...
    def prepare_store_kv(self, metadata: OffloadingConnectorMetadata): ...
    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]: ...
    def get_kv_connector_stats(self) -> KVConnectorStats | None: ...

class OffloadPromMetrics(KVConnectorPromMetrics):
    histogram_transfer_size: dict[tuple[int, str], PromMetricT]
    counter_kv_bytes: dict[tuple[int, str], PromMetricT]
    counter_kv_transfer_time: dict[tuple[int, str], PromMetricT]
    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> None: ...
    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0): ...
