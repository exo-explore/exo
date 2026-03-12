import zmq
from _typeshed import Incomplete
from contextlib import contextmanager
from dataclasses import dataclass, field
from torch.distributed import ProcessGroup
from typing import Any
from vllm.distributed.utils import (
    StatelessProcessGroup as StatelessProcessGroup,
    sched_yield as sched_yield,
)
from vllm.logger import init_logger as init_logger
from vllm.platforms import current_platform as current_platform
from vllm.utils.network_utils import (
    get_ip as get_ip,
    get_open_port as get_open_port,
    get_open_zmq_inproc_path as get_open_zmq_inproc_path,
    get_open_zmq_ipc_path as get_open_zmq_ipc_path,
    is_valid_ipv6_address as is_valid_ipv6_address,
)

VLLM_RINGBUFFER_WARNING_INTERVAL: Incomplete
from_bytes_big: Incomplete

def memory_fence() -> None: ...
def to_bytes_big(value: int, size: int) -> bytes: ...

logger: Incomplete
LONG_WAIT_TIME_LOG_MSG: str

class SpinCondition:
    is_reader: Incomplete
    last_read: Incomplete
    busy_loop_s: Incomplete
    local_notify_socket: zmq.Socket
    write_cancel_socket: zmq.Socket
    read_cancel_socket: zmq.Socket
    poller: Incomplete
    def __init__(
        self,
        is_reader: bool,
        context: zmq.Context,
        notify_address: str,
        busy_loop_s: float = 1,
    ) -> None: ...
    def record_read(self) -> None: ...
    def cancel(self) -> None: ...
    def wait(self, timeout_ms: int | None = None) -> None: ...
    def notify(self) -> None: ...

class ShmRingBuffer:
    n_reader: Incomplete
    metadata_size: Incomplete
    max_chunk_bytes: Incomplete
    max_chunks: Incomplete
    total_bytes_of_buffer: Incomplete
    data_offset: int
    metadata_offset: Incomplete
    is_creator: bool
    shared_memory: Incomplete
    def __init__(
        self,
        n_reader: int,
        max_chunk_bytes: int,
        max_chunks: int,
        name: str | None = None,
    ) -> None: ...
    def handle(self): ...
    def __reduce__(self): ...
    def __del__(self) -> None: ...
    @contextmanager
    def get_data(self, current_idx: int): ...
    @contextmanager
    def get_metadata(self, current_idx: int): ...

@dataclass
class Handle:
    local_reader_ranks: list[int] = field(default_factory=list)
    buffer_handle: tuple[int, int, int, str] | None = ...
    local_subscribe_addr: str | None = ...
    local_notify_addr: str | None = ...
    remote_subscribe_addr: str | None = ...
    remote_addr_ipv6: bool = ...

class MessageQueue:
    n_local_reader: Incomplete
    n_remote_reader: Incomplete
    shutting_down: bool
    buffer: Incomplete
    local_socket: Incomplete
    current_idx: int
    remote_socket: Incomplete
    local_reader_rank: int
    handle: Incomplete
    def __init__(
        self,
        n_reader,
        n_local_reader,
        local_reader_ranks: list[int] | None = None,
        max_chunk_bytes: int = ...,
        max_chunks: int = 10,
        connect_ip: str | None = None,
    ) -> None: ...
    def export_handle(self) -> Handle: ...
    @staticmethod
    def create_from_handle(handle: Handle, rank) -> MessageQueue: ...
    def wait_until_ready(self) -> None: ...
    def shutdown(self) -> None: ...
    @contextmanager
    def acquire_write(self, timeout: float | None = None): ...
    class ReadTimeoutWithWarnings:
        started: Incomplete
        deadline: Incomplete
        warning_wait_time_ms: int | None
        n_warning: int
        timeout: Incomplete
        def __init__(self, timeout: float | None, should_warn: bool) -> None: ...
        def timeout_ms(self) -> int | None: ...
        def should_warn(self) -> bool: ...

    @contextmanager
    def acquire_read(self, timeout: float | None = None, indefinite: bool = False): ...
    def enqueue(self, obj, timeout: float | None = None): ...
    def dequeue(self, timeout: float | None = None, indefinite: bool = False): ...
    @staticmethod
    def recv(socket: zmq.Socket, timeout: float | None) -> Any: ...
    def broadcast_object(self, obj=None): ...
    @staticmethod
    def create_from_process_group_single_reader(
        pg: ProcessGroup,
        max_chunk_bytes,
        max_chunks,
        reader_rank: int = 0,
        blocking: bool = False,
    ) -> tuple["MessageQueue", list[Handle]]: ...
    @staticmethod
    def create_from_process_group(
        pg: ProcessGroup | StatelessProcessGroup,
        max_chunk_bytes,
        max_chunks,
        writer_rank: int = 0,
        external_writer_handle=None,
        blocking: bool = True,
    ) -> MessageQueue: ...
