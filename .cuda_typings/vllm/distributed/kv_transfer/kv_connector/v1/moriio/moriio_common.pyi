import contextlib
import msgspec
import time
import torch
import zmq
from _typeshed import Incomplete
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from vllm import envs as envs
from vllm.config import VllmConfig as VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata as KVConnectorMetadata,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger as init_logger
from vllm.utils.network_utils import (
    get_ip as get_ip,
    get_open_port as get_open_port,
    make_zmq_socket as make_zmq_socket,
)

logger: Incomplete
Transfer = tuple[int, float]
EngineId = str
ReqId = str

@dataclass
class WriteTask:
    request_id: str
    dst_engine_id: str
    local_block_ids: list[int]
    remote_block_ids_hint: list[int] | None
    layer_name: str
    event: torch.cuda.Event
    remote_notify_port: int
    remote_ip: str
    enqueue_time: float = field(default_factory=time.perf_counter)
    retried: int = ...

@dataclass
class LayerTransferPlan:
    request_id: str
    layer_name: str
    sess_idx: int
    transfer_local_offsets: list[int]
    transfer_remote_offsets: list[int]
    transfer_sizes: list[int]
    use_batch: bool = ...

@dataclass
class RemoteAllocInfo:
    block_ids: list[int]
    writes_done: int = ...
    decode_dp_rank: int = ...
    transfer_offset: tuple[list[int], list[int], list[int]] | None = ...

class ROLE(Enum):
    PRODUCER = "producer"
    CONSUMER = "consumer"
    NOTINIT = "notinit"

class MoRIIOAgentMetadata(msgspec.Struct, omit_defaults=True, dict=True):
    engine_id: str
    agent_metadata: bytes
    kv_caches_base_addr: list[int]
    num_blocks: int
    block_len: int
    attn_backend_name: str

class RoleManager:
    def __init__(self) -> None: ...
    @classmethod
    def get_instance(cls) -> RoleManager: ...
    def set_role(self, role: ROLE) -> None: ...
    def get_role(self) -> ROLE: ...

def set_role(role: ROLE): ...
def get_role() -> ROLE: ...

class MoRIIOMode(Enum):
    READ = "read"
    WRITE = "write"

class MoRIIOError(Exception): ...
class HandshakeError(MoRIIOError): ...
class TransferError(MoRIIOError): ...

def get_moriio_mode() -> MoRIIOMode: ...
def get_port_offset(dp_rank: int, tp_rank: int, tp_size: int = 1) -> int: ...
@dataclass
class MoRIIOConfig:
    local_ip: str
    local_kv_port: int
    proxy_ip: str
    local_ping_port: int
    proxy_ping_port: int
    http_port: int
    handshake_port: int
    notify_port: int
    tp_rank: int
    dp_rank: int
    dp_size: int
    tp_size: int
    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig) -> MoRIIOConfig: ...

class MoRIIOConstants:
    GET_META_MSG: bytes
    POP_DONE_RECV: bytes
    OVER: bytes
    COMPLETION_PREFIX: str
    PING_INTERVAL: int
    MAX_PING_RETRIES: int
    DEFAULT_HANDSHAKE_PORT: str
    DEFAULT_NOTIFY_PORT: str
    VLLM_MORI_READ_ABORT_REQUEST_TIMEOUT: int

@dataclass
class ReqMeta:
    local_block_ids: list[int]
    remote_block_ids: list[int]
    remote_host: str
    remote_port: int
    remote_handshake_port: int
    remote_notify_port: int
    remote_engine_id: str
    tp_size: int
    remote_dp_size: int

class MoRIIOConnectorMetadata(KVConnectorMetadata):
    reqs_to_recv: dict[ReqId, ReqMeta]
    reqs_to_save: dict[ReqId, ReqMeta]
    reqs_to_send: dict[ReqId, float]
    def __init__(self) -> None: ...
    def add_new_req(
        self,
        request_id: ReqId,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
        write_mode: bool = False,
    ): ...

@contextlib.contextmanager
def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Socket]: ...
