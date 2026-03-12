import threading
import torch
import zmq
from _typeshed import Incomplete
from mori.io import IOEngine as IOEngine
from typing import Any
from vllm import envs as envs
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    HandshakeError as HandshakeError,
    LayerTransferPlan as LayerTransferPlan,
    MoRIIOAgentMetadata as MoRIIOAgentMetadata,
    MoRIIOConstants as MoRIIOConstants,
    MoRIIOError as MoRIIOError,
    ROLE as ROLE,
    RemoteAllocInfo as RemoteAllocInfo,
    TransferError as TransferError,
    WriteTask as WriteTask,
    get_port_offset as get_port_offset,
    get_role as get_role,
    zmq_ctx as zmq_ctx,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorWorker as MoRIIOConnectorWorker,
)
from vllm.logger import init_logger as init_logger
from vllm.utils.network_utils import (
    make_zmq_path as make_zmq_path,
    make_zmq_socket as make_zmq_socket,
)

logger: Incomplete

class MoRIIOWriter:
    def __init__(self, worker: MoRIIOConnectorWorker) -> None: ...
    @property
    def worker(self) -> MoRIIOConnectorWorker: ...
    def ensure_worker_started(self) -> None: ...
    def schedule_write(self, task: WriteTask) -> None: ...

class MoRIIOWrapper:
    tp_rank: Incomplete
    dp_rank: Incomplete
    moriio_engine: Incomplete
    remote_memory_metadata: Incomplete
    local_memory_registered: bool
    local_memory_metadata: Incomplete
    transfer_status: list[Any]
    remote_engine_ip: str | None
    notify_port: int | None
    lock: Incomplete
    done_req_ids: list[str]
    done_remote_allocate_req_dict: dict[str, RemoteAllocInfo]
    done_write_cache_req_ids: list[str]
    notify_thread: threading.Thread | None
    sessions: list[IOEngine.Session]
    paths: dict[str, zmq.Socket]
    def __init__(
        self, moriio_engine: IOEngine | None = None, tp_rank: int = 0, dp_rank: int = 0
    ) -> None: ...
    def set_moriio_engine(self, moriio_engine) -> None: ...
    def set_backend_type(self, backend_type) -> None: ...
    def get_agent_metadata(self): ...
    def register_remote_engine(self, remote_packed_engine_metadata): ...
    def register_local_tensor(self, tensor: torch.Tensor): ...
    def get_unpack_memory_metadata(self, packed_memory_metadata): ...
    def build_session(self, local_memory_metadata, remote_memory_metadata): ...
    def read_remote_data(
        self,
        transfer_size_byte,
        local_offset: int = 0,
        remote_offset: int = 0,
        session=None,
    ): ...
    def write_remote_data(
        self,
        transfer_size_byte,
        local_offset: int = 0,
        remote_offset: int = 0,
        session=None,
    ) -> None: ...
    def write_remote_data_single(
        self,
        transfer_size_byte,
        local_offset: int = 0,
        remote_offset: int = 0,
        sess_idx: int = 0,
    ) -> None: ...
    def waiting_for_transfer_complete(self) -> None: ...
    def async_wait_reqid(self) -> None: ...
    def send_notify(self, req_ids, remote_ip, remote_port) -> None: ...
    def pop_finished_req_ids(self): ...
    def pop_finished_write_req_ids(self): ...
    def shutdown(self) -> None: ...
