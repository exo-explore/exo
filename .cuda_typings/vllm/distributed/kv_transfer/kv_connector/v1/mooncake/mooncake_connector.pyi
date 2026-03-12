import asyncio
import msgspec
import torch
import zmq.asyncio
from _typeshed import Incomplete
from dataclasses import dataclass
from enum import IntEnum
from typing import Any
from vllm import envs as envs
from vllm.config import VllmConfig as VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import (
    EngineId as EngineId,
    TpKVTopology as TpKVTopology,
    get_current_attn_backend as get_current_attn_backend,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1 as KVConnectorBase_V1,
    KVConnectorMetadata as KVConnectorMetadata,
    KVConnectorRole as KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
    MooncakeBootstrapServer as MooncakeBootstrapServer,
    RegisterWorkerPayload as RegisterWorkerPayload,
)
from vllm.distributed.parallel_state import (
    get_pp_group as get_pp_group,
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
    is_local_first_rank as is_local_first_rank,
)
from vllm.forward_context import ForwardContext as ForwardContext
from vllm.logger import init_logger as init_logger
from vllm.utils.network_utils import (
    get_ip as get_ip,
    make_zmq_path as make_zmq_path,
    make_zmq_socket as make_zmq_socket,
)
from vllm.v1.attention.backend import AttentionMetadata as AttentionMetadata
from vllm.v1.attention.backends.utils import get_kv_cache_layout as get_kv_cache_layout
from vllm.v1.core.kv_cache_manager import KVCacheBlocks as KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput as SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig as KVCacheConfig
from vllm.v1.request import Request as Request, RequestStatus as RequestStatus

ReqId = str
TransferId = str
logger: Incomplete

class MooncakeXferMetadata(msgspec.Struct, omit_defaults=True):
    remote_hostname: str
    remote_port: int
    remote_tp_size: int
    remote_tp_rank: int
    req_blocks: dict[ReqId, tuple[TransferId, list[int]]]
    kv_caches_base_addr: list[int]

class MooncakeXferResponseStatus(IntEnum):
    FINISH = 0
    CONTINUE = 1
    ERROR = 2

class MooncakeXferResponse(msgspec.Struct, omit_defaults=True):
    status: MooncakeXferResponseStatus
    ok_reqs: list[ReqId] | None = ...
    err_reqs: list[ReqId] | None = ...
    err_msg: str | None = ...

@dataclass
class PullReqMeta:
    d_req_id: ReqId
    transfer_id: TransferId
    local_block_ids: list[int]
    remote_engine_id: EngineId
    remote_bootstrap_addr: str
    expire_time: float = ...
    pull_tasks_count: int = ...

@dataclass
class SendBlockMeta:
    p_req_id: ReqId
    transfer_id: TransferId
    local_block_ids: list[int]
    ready: asyncio.Event
    expire_time: float = ...
    need_send: int = ...
    sent: int = ...
    sending: int = ...

class MooncakeConnectorMetadata(KVConnectorMetadata):
    reqs_to_recv: dict[EngineId, dict[ReqId, PullReqMeta]]
    reqs_to_send: dict[ReqId, tuple[TransferId, list[int]]]
    reqs_not_processed: set[TransferId]
    def __init__(self) -> None: ...
    def add_new_req(
        self,
        request_id: ReqId,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
        load_remote_cache: bool = True,
    ): ...

class MooncakeConnector(KVConnectorBase_V1):
    engine_id: EngineId
    connector_scheduler: MooncakeConnectorScheduler | None
    connector_worker: MooncakeConnectorWorker | None
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
    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]: ...
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

class MooncakeConnectorScheduler:
    vllm_config: Incomplete
    is_kv_producer: bool
    is_kv_consumer: bool
    def __init__(self, vllm_config: VllmConfig, engine_id: str) -> None: ...
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

class MooncakeConnectorWorker:
    vllm_config: Incomplete
    engine: Incomplete
    hostname: Incomplete
    is_kv_producer: bool
    is_kv_consumer: bool
    num_sender_workers: Incomplete
    num_sender_tasks: Incomplete
    rpc_port: Incomplete
    side_channel_port: int
    engine_id: EngineId
    tp_rank: Incomplete
    tp_size: Incomplete
    num_blocks: int
    dp_rank: Incomplete
    pp_rank: Incomplete
    kv_caches_base_addr: list[int]
    device_kv_caches: dict[str, torch.Tensor]
    reqs_need_send: dict[TransferId, SendBlockMeta]
    sender_worker_queue: Incomplete
    sender_loop: Incomplete
    bootstrap_server: Incomplete
    receiver_loop: Incomplete
    finished_sending_reqs: set[ReqId]
    finished_recving_reqs: set[ReqId]
    block_size: Incomplete
    model_config: Incomplete
    cache_config: Incomplete
    use_mla: Incomplete
    backend_name: Incomplete
    kv_cache_layout: Incomplete
    kv_topo: Incomplete
    async_zmq_ctx: Incomplete
    def __init__(self, vllm_config: VllmConfig, engine_id: str) -> None: ...
    def __del__(self) -> None: ...
    def shutdown(self) -> None: ...
    async def register_worker_with_bootstrap(self) -> None: ...
    async def send_kv_to_decode(
        self, identity: bytes, sock: zmq.asyncio.Socket, meta: MooncakeXferMetadata
    ): ...
    def resolve_need_send(
        self, send_meta: SendBlockMeta, remote_tp_ranks: list[int]
    ): ...
    block_len: Incomplete
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]): ...
    async def fetch_finished_recving_reqs(self) -> set[ReqId]: ...
    async def fetch_finished_sending_reqs(self) -> set[ReqId]: ...
    def get_finished(self) -> tuple[set[str] | None, set[str] | None]: ...
    async def receive_kv_from_single_worker(
        self, worker_addr: str, pull_metas: dict[ReqId, PullReqMeta]
    ): ...
    def process_pulling_result(
        self, response: MooncakeXferResponse, pull_metas: dict[ReqId, PullReqMeta]
    ): ...
    def receive_kv(
        self, remote_engine_id: EngineId, pull_metas: dict[ReqId, PullReqMeta]
    ): ...
    async def handle_new_engine_id(
        self, remote_engine_id: EngineId, pull_metas: dict[ReqId, PullReqMeta]
    ): ...
    async def record_send_reqs(self, metadata: MooncakeConnectorMetadata): ...
    def start_load_kv(self, metadata: MooncakeConnectorMetadata): ...

def group_concurrent_contiguous(
    src_indices: list[int], dst_indices: list[int]
) -> tuple[list[list[int]], list[list[int]]]: ...
def get_mooncake_side_channel_port(vllm_config: VllmConfig) -> int: ...
def should_launch_bootstrap_server(vllm_config: VllmConfig) -> bool: ...
def get_mooncake_bootstrap_addr(vllm_config: VllmConfig) -> tuple[str, int]: ...
