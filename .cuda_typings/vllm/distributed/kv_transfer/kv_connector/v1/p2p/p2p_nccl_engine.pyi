import torch
from _typeshed import Incomplete
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from vllm.config.kv_transfer import KVTransferConfig as KVTransferConfig
from vllm.distributed.device_communicators.pynccl_wrapper import (
    NCCLLibrary as NCCLLibrary,
    buffer_type as buffer_type,
    cudaStream_t as cudaStream_t,
    ncclComm_t as ncclComm_t,
    ncclDataTypeEnum as ncclDataTypeEnum,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.tensor_memory_pool import (
    TensorMemoryPool as TensorMemoryPool,
)
from vllm.utils.network_utils import get_ip as get_ip
from vllm.utils.torch_utils import current_stream as current_stream

logger: Incomplete
DEFAULT_MEM_POOL_SIZE_GB: int

@contextmanager
def set_p2p_nccl_context(num_channels: str): ...
@dataclass
class SendQueueItem:
    tensor_id: str
    remote_address: str
    tensor: torch.Tensor

class P2pNcclEngine:
    config: Incomplete
    rank: Incomplete
    local_rank: Incomplete
    device: Incomplete
    nccl: Incomplete
    zmq_address: Incomplete
    proxy_address: str
    http_address: str
    context: Incomplete
    router_socket: Incomplete
    poller: Incomplete
    send_store_cv: Incomplete
    send_queue_cv: Incomplete
    recv_store_cv: Incomplete
    send_stream: Incomplete
    recv_stream: Incomplete
    pool: Incomplete
    send_type: Incomplete
    send_store: dict[str, torch.Tensor]
    send_queue: deque[SendQueueItem]
    recv_store: dict[str, Any]
    recv_request_id_to_tensor_ids: dict[str, set[str]]
    send_request_id_to_tensor_ids: dict[str, set[str]]
    socks: dict[str, Any]
    comms: dict[str, Any]
    buffer_size: int
    buffer_size_threshold: Incomplete
    nccl_num_channels: Incomplete
    def __init__(
        self,
        local_rank: int,
        config: KVTransferConfig,
        hostname: str = "",
        port_offset: int = 0,
        library_path: str | None = None,
    ) -> None: ...
    def create_connect(self, remote_address: str | None = None): ...
    def send_tensor(
        self, tensor_id: str, tensor: torch.Tensor, remote_address: str | None = None
    ) -> bool: ...
    def recv_tensor(
        self, tensor_id: str, remote_address: str | None = None
    ) -> torch.Tensor: ...
    def listen_for_requests(self) -> None: ...
    def have_sent_tensor_id(self, tensor_id: str): ...
    def have_received_tensor_id(self, tensor_id: str): ...
    def send_async(self) -> None: ...
    def wait_for_sent(self) -> None: ...
    def send_sync(self, item: SendQueueItem) -> bool: ...
    def get_finished(
        self, finished_req_ids: set[str], no_compile_layers
    ) -> tuple[set[str] | None, set[str] | None]: ...
    def ping(self) -> None: ...
    def send(self, comm, tensor: torch.Tensor, dst: int, stream=None): ...
    def recv(self, comm, tensor: torch.Tensor, src: int, stream=None): ...
    def close(self) -> None: ...
