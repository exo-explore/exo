import abc
import msgspec
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Callable as Callable
from typing import Any
from vllm.config.kv_events import KVEventsConfig as KVEventsConfig
from vllm.logger import init_logger as init_logger
from vllm.v1.core.kv_cache_utils import ExternalBlockHash as ExternalBlockHash

logger: Incomplete

class EventBatch(msgspec.Struct, array_like=True, omit_defaults=True, gc=False):
    ts: float
    events: list[Any]
    data_parallel_rank: int | None = ...

class KVCacheEvent(
    msgspec.Struct, array_like=True, omit_defaults=True, gc=False, tag=True
): ...

MEDIUM_GPU: str

class BlockStored(KVCacheEvent):
    block_hashes: list[ExternalBlockHash]
    parent_block_hash: ExternalBlockHash | None
    token_ids: list[int]
    block_size: int
    lora_id: int | None
    medium: str | None
    lora_name: str | None
    extra_keys: list[tuple[Any, ...] | None] | None = ...
    def __hash__(self) -> int: ...

class BlockRemoved(KVCacheEvent):
    block_hashes: list[ExternalBlockHash]
    medium: str | None
    def __hash__(self) -> int: ...

class AllBlocksCleared(KVCacheEvent): ...

class KVEventBatch(EventBatch):
    events: list[BlockStored | BlockRemoved | AllBlocksCleared]

class KVEventAggregator:
    def __init__(self, num_workers: int) -> None: ...
    def add_events(self, events: list[KVCacheEvent]) -> None: ...
    def get_common_events(self) -> list[KVCacheEvent]: ...
    def get_all_events(self) -> list[KVCacheEvent]: ...
    def clear_events(self) -> None: ...
    def increment_workers(self, count: int = 1) -> None: ...
    def reset_workers(self) -> None: ...
    def get_number_of_workers(self) -> int: ...

class KVConnectorKVEvents(ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    def add_events(self, events: list[KVCacheEvent]) -> None: ...
    @abstractmethod
    def aggregate(self) -> KVConnectorKVEvents: ...
    @abstractmethod
    def increment_workers(self, count: int = 1) -> None: ...
    @abstractmethod
    def get_all_events(self) -> list[KVCacheEvent]: ...
    @abstractmethod
    def get_number_of_workers(self) -> int: ...
    @abstractmethod
    def clear_events(self) -> None: ...
    def merge(self, other: KVConnectorKVEvents) -> KVConnectorKVEvents: ...

class EventPublisher(ABC, metaclass=abc.ABCMeta):
    def __init__(self, data_parallel_rank: int = 0) -> None: ...
    @abstractmethod
    def publish(self, events: EventBatch) -> None: ...
    @abstractmethod
    def shutdown(self) -> None: ...

class NullEventPublisher(EventPublisher):
    def publish(self, events) -> None: ...
    def shutdown(self) -> None: ...

class ZmqEventPublisher(EventPublisher):
    SHUTDOWN_TIMEOUT: float
    END_SEQ: Incomplete
    def __init__(
        self,
        data_parallel_rank: int,
        endpoint: str = "tcp://*:5557",
        replay_endpoint: str | None = None,
        buffer_steps: int = 10000,
        hwm: int = 100000,
        max_queue_size: int = 100000,
        topic: str = "",
    ) -> None: ...
    def publish(self, events: EventBatch) -> None: ...
    def shutdown(self) -> None: ...
    @staticmethod
    def offset_endpoint_port(
        endpoint: str | None, data_parallel_rank: int
    ) -> str | None: ...

class EventPublisherFactory:
    @classmethod
    def register_publisher(
        cls, name: str, ctor: Callable[..., EventPublisher]
    ) -> None: ...
    @classmethod
    def create(
        cls, config: KVEventsConfig | None, data_parallel_rank: int = 0
    ) -> EventPublisher: ...
