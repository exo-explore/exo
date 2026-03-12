import torch
import zmq
from _typeshed import Incomplete
from collections import defaultdict
from typing import Any
from vllm.config import VllmConfig as VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1 as KVConnectorBase_V1,
    KVConnectorMetadata as KVConnectorMetadata,
    KVConnectorRole as KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    EngineId as EngineId,
    HandshakeError as HandshakeError,
    MoRIIOAgentMetadata as MoRIIOAgentMetadata,
    MoRIIOConfig as MoRIIOConfig,
    MoRIIOConnectorMetadata as MoRIIOConnectorMetadata,
    MoRIIOConstants as MoRIIOConstants,
    MoRIIOMode as MoRIIOMode,
    ROLE as ROLE,
    ReqId as ReqId,
    ReqMeta as ReqMeta,
    WriteTask as WriteTask,
    get_moriio_mode as get_moriio_mode,
    get_port_offset as get_port_offset,
    get_role as get_role,
    set_role as set_role,
    zmq_ctx as zmq_ctx,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine import (
    MoRIIOWrapper as MoRIIOWrapper,
    MoRIIOWriter as MoRIIOWriter,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
    get_tp_group as get_tp_group,
    get_world_group as get_world_group,
)
from vllm.forward_context import ForwardContext as ForwardContext
from vllm.logger import init_logger as init_logger
from vllm.utils.network_utils import (
    get_ip as get_ip,
    make_zmq_path as make_zmq_path,
    make_zmq_socket as make_zmq_socket,
)
from vllm.v1.attention.backend import AttentionMetadata as AttentionMetadata
from vllm.v1.attention.selector import get_attn_backend as get_attn_backend
from vllm.v1.core.kv_cache_manager import KVCacheBlocks as KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput as SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig as KVCacheConfig
from vllm.v1.request import Request as Request, RequestStatus as RequestStatus

logger: Incomplete
MoRIIO_enabled: bool

def is_moriio_available() -> bool: ...

class MoRIIOConnector(KVConnectorBase_V1):
    kv_transfer_config: Incomplete
    engine_id: Incomplete
    mode: Incomplete
    connector_scheduler: MoRIIOConnectorScheduler | None
    connector_worker: MoRIIOConnectorWorker | None
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
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
        self, request: Request, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]: ...
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]): ...
    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]: ...
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
    def has_connector_metadata(self) -> bool: ...

class MoRIIOConnectorScheduler:
    vllm_config: Incomplete
    kv_transfer_config: Incomplete
    block_size: Incomplete
    engine_id: EngineId
    mode: Incomplete
    host_ip: Incomplete
    handshake_port: Incomplete
    side_notify_port: Incomplete
    tp_size: Incomplete
    dp_rank: Incomplete
    is_producer: Incomplete
    paths: dict[str, zmq.Socket]
    def __init__(self, vllm_config: VllmConfig, engine_id: str) -> None: ...
    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int, bool]: ...
    def send_notify_block(
        self, req_id: str, block_notify_list: list[int], host=None, port=None
    ): ...
    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
        connector_worker: MoRIIOConnectorWorker | None = None,
    ): ...
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata: ...
    def shutdown(self) -> None: ...
    def request_finished(
        self, request: Request, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]: ...

class MoRIIOConnectorWorker:
    moriio_config: Incomplete
    mode: Incomplete
    vllm_config: Incomplete
    kv_transfer_config: Incomplete
    is_producer: Incomplete
    tp_rank: Incomplete
    dp_rank: Incomplete
    local_ip: Incomplete
    local_kv_port: Incomplete
    proxy_ip: Incomplete
    local_ping_port: Incomplete
    proxy_ping_port: Incomplete
    http_port: Incomplete
    handshake_port: Incomplete
    notify_port: Incomplete
    zmq_context: Incomplete
    metadata_address: Incomplete
    request_address: Incomplete
    moriio_engine: Incomplete
    moriio_wrapper: Incomplete
    local_kv_cache_metadata: list[bytes]
    local_kv_cache_size: list[int]
    layer_name_to_local_kv_cache_metadata: dict[str, list[bytes]]
    remote_kv_cache_metadata: list[bytes]
    remote_kv_cache_size: list[int]
    layer_name_to_remote_kv_cache_metadata: dict[str, dict[str, list[Any]]]
    remote_moriio_metadata: dict[EngineId, MoRIIOAgentMetadata]
    slot_size_bytes: int
    load_ready_flag: dict[str, bool]
    write_ready_flags: dict[str, bool]
    kv_cache_shape: Incomplete
    block_shape: Incomplete
    kv_element_size: int
    side_channel_port: int
    engine_id: EngineId
    world_size: Incomplete
    tp_group: Incomplete
    kv_caches: dict[str, torch.Tensor]
    kv_caches_base_addr: dict[EngineId, list[int]]
    num_regions: int
    num_layers: int
    dst_num_blocks: dict[EngineId, int]
    block_size: Incomplete
    model_config: Incomplete
    cache_config: Incomplete
    block_window_per_layer: list[int | None]
    use_mla: Incomplete
    built_session: bool
    built_write_session: defaultdict[str, list]
    backend_name: Incomplete
    def __init__(self, vllm_config: VllmConfig, engine_id: str) -> None: ...
    def schedule_write_blocks(
        self,
        request_id: str,
        dst_engine_id: str,
        local_block_ids: list[int],
        remote_block_ids: list[int] | None,
        layer_name: str,
        kv_layer: torch.Tensor,
        remote_notify_port: int,
        remote_ip: str,
    ) -> None: ...
    def shutdown(self) -> None: ...
    def __del__(self) -> None: ...
    num_blocks: Incomplete
    block_len: Incomplete
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]): ...
    def get_finished(self) -> tuple[set[str], set[str]]: ...
    def save_kv_layer(
        self,
        metadata: MoRIIOConnectorMetadata,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ): ...
    def get_engine_name_with_dp(self, engine_name, dp_rank): ...
    def start_load_kv(self, metadata: MoRIIOConnectorMetadata): ...
    def merge_contiguous_blocks(
        self,
        offsets_local: list[int],
        offsets_remote: list[int],
        sizes: list[int],
        assume_sorted: bool = False,
    ) -> tuple[list[int], list[int], list[int]]: ...
