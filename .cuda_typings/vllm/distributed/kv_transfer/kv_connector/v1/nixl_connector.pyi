import contextlib
import numpy as np
import torch
import zmq
from _typeshed import Incomplete
from collections.abc import Iterator
from dataclasses import dataclass
from rixl._bindings import nixlXferTelemetry
from typing import Any
from vllm import envs as envs
from vllm.config import VllmConfig as VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import (
    BlockIds as BlockIds,
    EngineId as EngineId,
    TpKVTopology as TpKVTopology,
    get_current_attn_backend as get_current_attn_backend,
    get_current_attn_backends as get_current_attn_backends,
    kv_postprocess_blksize_and_layout_on_receive as kv_postprocess_blksize_and_layout_on_receive,
    kv_postprocess_blksize_on_receive as kv_postprocess_blksize_on_receive,
    kv_postprocess_layout_on_receive as kv_postprocess_layout_on_receive,
    yield_req_data as yield_req_data,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    CopyBlocksOp as CopyBlocksOp,
    KVConnectorBase_V1 as KVConnectorBase_V1,
    KVConnectorHandshakeMetadata as KVConnectorHandshakeMetadata,
    KVConnectorMetadata as KVConnectorMetadata,
    KVConnectorRole as KVConnectorRole,
    SupportsHMA as SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics as KVConnectorPromMetrics,
    KVConnectorStats as KVConnectorStats,
    PromMetric as PromMetric,
    PromMetricT as PromMetricT,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
    get_tp_group as get_tp_group,
)
from vllm.forward_context import ForwardContext as ForwardContext
from vllm.logger import init_logger as init_logger
from vllm.platforms import current_platform as current_platform
from vllm.utils.math_utils import cdiv as cdiv
from vllm.utils.network_utils import (
    make_zmq_path as make_zmq_path,
    make_zmq_socket as make_zmq_socket,
)
from vllm.v1.attention.backend import (
    AttentionBackend as AttentionBackend,
    AttentionMetadata as AttentionMetadata,
)
from vllm.v1.attention.backends.utils import get_kv_cache_layout as get_kv_cache_layout
from vllm.v1.core.kv_cache_manager import KVCacheBlocks as KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput as SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec as FullAttentionSpec,
    KVCacheConfig as KVCacheConfig,
    MambaSpec as MambaSpec,
    SlidingWindowSpec as SlidingWindowSpec,
)
from vllm.v1.request import Request as Request
from vllm.v1.worker.block_table import BlockTable as BlockTable
from vllm.v1.worker.utils import select_common_block_size as select_common_block_size

TransferHandle = int
ReqId = str
NIXL_CONNECTOR_VERSION: int
GET_META_MSG: bytes
logger: Incomplete

@dataclass
class NixlAgentMetadata:
    engine_id: str
    agent_metadata: bytes
    kv_caches_base_addr: list[int]
    device_id: int
    num_blocks: int
    block_lens: list[int]
    kv_cache_layout: str
    block_size: int

@dataclass
class NixlHandshakePayload(KVConnectorHandshakeMetadata):
    compatibility_hash: str
    agent_metadata_bytes: bytes

def compute_nixl_compatibility_hash(
    vllm_config: VllmConfig, attn_backend_name: str, cross_layers_blocks: bool
) -> str: ...
@dataclass
class RemoteMeta:
    block_ids: BlockIds
    host: str
    port: int
    engine_id: str
    request_id: str

@dataclass
class ReqMeta:
    local_block_ids: BlockIds
    local_physical_block_ids: BlockIds
    tp_size: int
    remote: RemoteMeta | None = ...

class NixlConnectorMetadata(KVConnectorMetadata):
    reqs_to_recv: dict[ReqId, ReqMeta]
    reqs_to_save: dict[ReqId, ReqMeta]
    reqs_to_send: dict[ReqId, float]
    reqs_in_batch: set[ReqId]
    reqs_not_processed: set[ReqId]
    def __init__(self) -> None: ...
    def add_new_req_to_save(
        self,
        request_id: ReqId,
        local_block_ids: BlockIds,
        kv_transfer_params: dict[str, Any],
    ): ...
    def add_new_req_to_recv(
        self,
        request_id: ReqId,
        local_block_ids: BlockIds,
        kv_transfer_params: dict[str, Any],
    ): ...

class NixlConnector(KVConnectorBase_V1, SupportsHMA):
    @property
    def prefer_cross_layer_blocks(self) -> bool: ...
    engine_id: EngineId
    kv_transfer_config: Incomplete
    connector_scheduler: NixlConnectorScheduler | None
    connector_worker: NixlConnectorWorker | None
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ) -> None: ...
    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig): ...
    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]: ...
    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ): ...
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata: ...
    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]: ...
    def set_xfer_handshake_metadata(
        self, metadata: dict[int, KVConnectorHandshakeMetadata]
    ) -> None: ...
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]): ...
    def register_cross_layers_kv_cache(
        self, kv_cache: torch.Tensor, attn_backend: type[AttentionBackend]
    ): ...
    def set_host_xfer_buffer_ops(self, copy_operation: CopyBlocksOp): ...
    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]: ...
    def get_block_ids_with_load_errors(self) -> set[int]: ...
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
    def shutdown(self) -> None: ...
    def get_handshake_metadata(self) -> KVConnectorHandshakeMetadata | None: ...

class NixlConnectorScheduler:
    vllm_config: Incomplete
    block_size: Incomplete
    engine_id: EngineId
    kv_cache_config: Incomplete
    side_channel_host: Incomplete
    side_channel_port: Incomplete
    use_host_buffer: bool
    blocks_per_sw: Incomplete
    def __init__(
        self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: KVCacheConfig
    ) -> None: ...
    def shutdown(self) -> None: ...
    def get_sw_clipped_blocks(self, block_ids: BlockIds) -> BlockIds: ...
    def set_xfer_handshake_metadata(
        self, metadata: dict[int, KVConnectorHandshakeMetadata]
    ) -> None: ...
    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int, bool]: ...
    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ): ...
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata: ...
    def request_finished(
        self, request: Request, block_ids: BlockIds
    ) -> tuple[bool, dict[str, Any] | None]: ...

class NixlConnectorWorker:
    vllm_config: Incomplete
    block_size: int
    kv_transfer_config: Incomplete
    nixl_backends: Incomplete
    kv_cache_config: Incomplete
    nixl_wrapper: Incomplete
    engine_id: EngineId
    tp_rank: Incomplete
    world_size: Incomplete
    tp_group: Incomplete
    num_blocks: Incomplete
    enable_permute_local_kv: bool
    device_type: Incomplete
    kv_buffer_device: str
    device_kv_caches: dict[str, torch.Tensor]
    host_xfer_buffers: dict[str, torch.Tensor]
    use_host_buffer: bool
    nixl_memory_type: Incomplete
    copy_blocks: CopyBlocksOp | None
    device_id: int
    kv_caches_base_addr: Incomplete
    num_regions: int
    num_layers: int
    src_xfer_handles_by_block_size: dict[int, int]
    src_xfer_handles_by_tp_ratio: dict[int, list[int]]
    dst_xfer_side_handles: Incomplete
    dst_num_blocks: dict[EngineId, int]
    xfer_handshake_metadata: NixlHandshakePayload | None
    model_config: Incomplete
    cache_config: Incomplete
    use_mla: Incomplete
    attn_backend: Incomplete
    backend_name: Incomplete
    kv_cache_layout: Incomplete
    host_buffer_kv_cache_layout: Incomplete
    compat_hash: str | None
    kv_topo: TpKVTopology | None
    consumer_notification_counts_by_req: Incomplete
    xfer_stats: Incomplete
    enforce_compat_hash: Incomplete
    def __init__(
        self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: KVCacheConfig
    ) -> None: ...
    def initialize_host_xfer_buffer(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> None: ...
    def set_host_xfer_buffer_ops(self, copy_operation: CopyBlocksOp): ...
    block_len_per_layer: Incomplete
    seen_base_addresses: Incomplete
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]): ...
    def register_local_xfer_handler(
        self, block_size: int
    ) -> tuple[int, list[tuple[int, int, int]]]: ...
    def add_remote_agent(
        self,
        nixl_agent_meta: NixlAgentMetadata,
        remote_tp_rank: int = 0,
        remote_tp_size: int = 1,
    ) -> str: ...
    def sync_recved_kv_to_device(self, req_id: str, meta: ReqMeta): ...
    def save_kv_to_host(self, metadata: NixlConnectorMetadata): ...
    def post_process_device_kv_on_receive(
        self, block_size_ratio: int, block_ids_list: list[list[int]]
    ): ...
    def get_finished(self) -> tuple[set[str], set[str]]: ...
    def start_load_kv(self, metadata: NixlConnectorMetadata): ...
    def get_mapped_blocks(
        self, block_ids: np.ndarray, block_size_ratio: int
    ) -> np.ndarray: ...
    def get_backend_aware_kv_block_len(self, layer_idx: int) -> int: ...
    def get_kv_connector_stats(self) -> KVConnectorStats | None: ...
    def get_block_ids_with_load_errors(self) -> set[int]: ...
    def __del__(self) -> None: ...
    def shutdown(self) -> None: ...

@contextlib.contextmanager
def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Socket]: ...
@dataclass
class NixlKVConnectorStats(KVConnectorStats):
    def __post_init__(self) -> None: ...
    data: dict[str, list[float | int]] = ...
    def reset(self) -> None: ...
    def record_transfer(self, res: nixlXferTelemetry): ...
    def record_failed_transfer(self) -> None: ...
    def record_failed_notification(self) -> None: ...
    def record_kv_expired_req(self) -> None: ...
    def clone_and_reset(self) -> NixlKVConnectorStats: ...
    def is_empty(self) -> bool: ...
    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats: ...
    def reduce(self) -> dict[str, int | float]: ...
    @property
    def num_successful_transfers(self) -> int: ...

class NixlPromMetrics(KVConnectorPromMetrics):
    nixl_histogram_xfer_time: Incomplete
    nixl_histogram_post_time: Incomplete
    nixl_histogram_bytes_transferred: Incomplete
    nixl_histogram_num_descriptors: Incomplete
    counter_nixl_num_failed_transfers: Incomplete
    counter_nixl_num_failed_notifications: Incomplete
    counter_nixl_num_kv_expired_reqs: Incomplete
    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> None: ...
    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0): ...
