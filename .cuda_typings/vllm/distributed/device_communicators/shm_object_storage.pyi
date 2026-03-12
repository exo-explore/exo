import abc
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Callable as Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.synchronize import Lock as LockType
from typing import Any
from vllm.logger import init_logger as init_logger

logger: Incomplete

class SingleWriterShmRingBuffer:
    data_buffer_size: Incomplete
    is_writer: Incomplete
    ID_NBYTES: int
    ID_MAX: Incomplete
    SIZE_NBYTES: int
    MD_SIZE: Incomplete
    monotonic_id_end: int
    monotonic_id_start: int
    data_buffer_start: int
    data_buffer_end: int
    metadata: dict[int, int]
    shared_memory: Incomplete
    def __init__(
        self, data_buffer_size: int, name: str | None = None, create: bool = False
    ) -> None: ...
    def handle(self): ...
    def clear(self) -> None: ...
    def close(self) -> None: ...
    def __del__(self) -> None: ...
    def int2byte(self, integer: int) -> bytes: ...
    def byte2int(self, byte_data: bytes) -> int: ...
    def allocate_buf(self, size: int) -> tuple[int, int]: ...
    @contextmanager
    def access_buf(self, address: int): ...
    def free_buf(
        self, is_free_fn: Callable[[int, memoryview], bool], nbytes: int | None = None
    ) -> Iterable[int]: ...

class ObjectSerde(ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    def serialize(self, value: Any) -> tuple[Any, int, bytes, int]: ...
    @abstractmethod
    def deserialize(self, data: memoryview) -> Any: ...

class MsgpackSerde(ObjectSerde):
    encoder: Incomplete
    tensor_decoder: Incomplete
    mm_decoder: Incomplete
    def __init__(self) -> None: ...
    def serialize(self, value: Any) -> tuple[bytes | list[bytes], int, bytes, int]: ...
    def deserialize(self, data_view: memoryview) -> Any: ...

@dataclass
class ShmObjectStorageHandle:
    max_object_size: int
    n_readers: int
    ring_buffer_handle: tuple[int, str]
    serde_class: type[ObjectSerde]
    reader_lock: LockType | None

class SingleWriterShmObjectStorage:
    max_object_size: Incomplete
    n_readers: Incomplete
    serde_class: Incomplete
    ser_de: Incomplete
    ring_buffer: Incomplete
    is_writer: Incomplete
    flag_bytes: int
    key_index: dict[str, tuple[int, int]]
    id_index: dict[int, str]
    writer_flag: dict[int, int]
    def __init__(
        self,
        max_object_size: int,
        n_readers: int,
        ring_buffer: SingleWriterShmRingBuffer,
        serde_class: type[ObjectSerde] = ...,
        reader_lock: LockType | None = None,
    ) -> None: ...
    def clear(self) -> None: ...
    def copy_to_buffer(
        self,
        data: bytes | list[bytes],
        data_bytes: int,
        metadata: bytes,
        md_bytes: int,
        data_view: memoryview,
    ) -> None: ...
    def increment_writer_flag(self, id: int) -> None: ...
    def increment_reader_flag(self, data_view: memoryview) -> None: ...
    def free_unused(self) -> None: ...
    def is_cached(self, key: str) -> bool: ...
    def get_cached(self, key: str) -> tuple[int, int]: ...
    def put(self, key: str, value: Any) -> tuple[int, int]: ...
    def get(self, address: int, monotonic_id: int) -> Any: ...
    def touch(self, key: str, address: int = 0, monotonic_id: int = 0) -> None: ...
    def close(self) -> None: ...
    def handle(self): ...
    @staticmethod
    def create_from_handle(
        handle: ShmObjectStorageHandle,
    ) -> SingleWriterShmObjectStorage: ...
    def default_is_free_check(self, id: int, buf: memoryview) -> bool: ...
