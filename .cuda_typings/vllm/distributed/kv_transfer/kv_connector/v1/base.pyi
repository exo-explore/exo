import abc
import enum
import torch
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any
from vllm.config import VllmConfig as VllmConfig
from vllm.distributed.kv_events import (
    KVCacheEvent as KVCacheEvent,
    KVConnectorKVEvents as KVConnectorKVEvents,
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

CopyBlocksOp: Incomplete
logger: Incomplete

class SupportsHMA(ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]: ...

def supports_hma(connector: Any) -> bool: ...

class KVConnectorRole(enum.Enum):
    SCHEDULER = 0
    WORKER = 1

class KVConnectorHandshakeMetadata(ABC): ...
class KVConnectorMetadata(ABC): ...

class KVConnectorBase_V1(ABC, metaclass=abc.ABCMeta):
    @property
    def prefer_cross_layer_blocks(self) -> bool: ...
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ) -> None: ...
    @property
    def role(self) -> KVConnectorRole: ...
    def bind_connector_metadata(
        self, connector_metadata: KVConnectorMetadata
    ) -> None: ...
    def clear_connector_metadata(self) -> None: ...
    def has_connector_metadata(self) -> bool: ...
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]): ...
    def register_cross_layers_kv_cache(
        self, kv_cache: torch.Tensor, attn_backend: type["AttentionBackend"]
    ): ...
    def set_host_xfer_buffer_ops(self, copy_operation: CopyBlocksOp): ...
    def handle_preemptions(self, preempted_req_ids: set[str]): ...
    @abstractmethod
    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None: ...
    @abstractmethod
    def wait_for_layer_load(self, layer_name: str) -> None: ...
    @abstractmethod
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None: ...
    @abstractmethod
    def wait_for_save(self): ...
    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]: ...
    def get_block_ids_with_load_errors(self) -> set[int]: ...
    def shutdown(self) -> None: ...
    def get_kv_connector_stats(self) -> KVConnectorStats | None: ...
    def get_kv_connector_kv_cache_events(self) -> KVConnectorKVEvents | None: ...
    def get_handshake_metadata(self) -> KVConnectorHandshakeMetadata | None: ...
    @abstractmethod
    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]: ...
    @abstractmethod
    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ): ...
    @abstractmethod
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
    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool: ...
    def get_finished_count(self) -> int | None: ...
    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> KVConnectorStats | None: ...
    def set_xfer_handshake_metadata(
        self, metadata: dict[int, KVConnectorHandshakeMetadata]
    ) -> None: ...
    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: VllmConfig,
        metric_types: dict[type["PromMetric"], type["PromMetricT"]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> KVConnectorPromMetrics | None: ...
    def reset_cache(self) -> bool | None: ...
