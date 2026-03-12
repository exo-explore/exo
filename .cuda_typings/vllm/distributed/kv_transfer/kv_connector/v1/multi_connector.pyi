import torch
from _typeshed import Incomplete
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from vllm.config import VllmConfig as VllmConfig
from vllm.config.kv_transfer import KVTransferConfig as KVTransferConfig
from vllm.distributed.kv_events import KVCacheEvent as KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.base import (
    KVConnectorBaseType as KVConnectorBaseType,
)
from vllm.distributed.kv_transfer.kv_connector.factory import (
    KVConnectorFactory as KVConnectorFactory,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    CopyBlocksOp as CopyBlocksOp,
    KVConnectorBase_V1 as KVConnectorBase_V1,
    KVConnectorHandshakeMetadata as KVConnectorHandshakeMetadata,
    KVConnectorMetadata as KVConnectorMetadata,
    KVConnectorRole as KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics as KVConnectorPromMetrics,
    KVConnectorStats as KVConnectorStats,
    PromMetric as PromMetric,
    PromMetricT as PromMetricT,
)
from vllm.forward_context import ForwardContext as ForwardContext
from vllm.logger import init_logger as init_logger
from vllm.v1.attention.backend import (
    AttentionBackend as AttentionBackend,
    AttentionMetadata as AttentionMetadata,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks as KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput as SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig as KVCacheConfig
from vllm.v1.outputs import KVConnectorOutput as KVConnectorOutput
from vllm.v1.request import Request as Request

logger: Incomplete

@dataclass
class MultiKVConnectorMetadata(KVConnectorMetadata):
    metadata: tuple[KVConnectorMetadata, ...]
    extra_async_saves: dict[str, int] | None = ...

@dataclass
class MultiKVConnectorStats(KVConnectorStats):
    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats: ...
    def reset(self) -> None: ...
    def reduce(self) -> dict[str, Any]: ...
    def is_empty(self) -> bool: ...
    def __getitem__(self, connector_id: str) -> KVConnectorStats: ...
    def __setitem__(self, connector_id: str, stats: KVConnectorStats): ...

class MultiKVConnectorPromMetrics(KVConnectorPromMetrics):
    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
        prom_metrics: dict[str, KVConnectorPromMetrics],
    ) -> None: ...
    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0): ...

class MultiConnector(KVConnectorBase_V1):
    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool: ...
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ) -> None: ...
    @property
    def prefer_cross_layer_blocks(self) -> bool: ...
    def register_cross_layers_kv_cache(
        self, kv_cache: torch.Tensor, attn_backend: type[AttentionBackend]
    ): ...
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]): ...
    def bind_connector_metadata(
        self, connector_metadata: KVConnectorMetadata
    ) -> None: ...
    def clear_connector_metadata(self) -> None: ...
    def shutdown(self) -> None: ...
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
    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]: ...
    def get_block_ids_with_load_errors(self) -> set[int]: ...
    def set_host_xfer_buffer_ops(self, copy_operation: CopyBlocksOp): ...
    def handle_preemptions(self, preempted_req_ids: set[str]): ...
    def get_finished_count(self) -> int | None: ...
    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]: ...
    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ): ...
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> MultiKVConnectorMetadata: ...
    def update_connector_output(self, connector_output: KVConnectorOutput): ...
    def get_handshake_metadata(self) -> KVConnectorHandshakeMetadata | None: ...
    def set_xfer_handshake_metadata(
        self, metadata: dict[int, KVConnectorHandshakeMetadata]
    ) -> None: ...
    def request_finished(
        self, request: Request, blocks: list[int]
    ) -> tuple[bool, dict[str, Any] | None]: ...
    def take_events(self) -> Iterable["KVCacheEvent"]: ...
    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig) -> str | None: ...
    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> KVConnectorStats | None: ...
    def get_kv_connector_stats(self) -> MultiKVConnectorStats | None: ...
    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: VllmConfig,
        metric_types: dict[type["PromMetric"], type["PromMetricT"]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> KVConnectorPromMetrics: ...
    def reset_cache(self) -> bool: ...
