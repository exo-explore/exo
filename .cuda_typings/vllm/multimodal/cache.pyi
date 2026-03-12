import abc
from .inputs import (
    MultiModalBatchedField as MultiModalBatchedField,
    MultiModalFeatureSpec as MultiModalFeatureSpec,
    MultiModalFieldElem as MultiModalFieldElem,
    MultiModalKwargsItem as MultiModalKwargsItem,
    MultiModalKwargsItems as MultiModalKwargsItems,
    NestedTensors as NestedTensors,
)
from .processing.processor import ResolvedPromptUpdate as ResolvedPromptUpdate
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from multiprocessing.synchronize import Lock as LockType
from typing import Generic, TypeAlias
from typing_extensions import override
from vllm.config import ModelConfig as ModelConfig, VllmConfig as VllmConfig
from vllm.distributed.device_communicators.shm_object_storage import (
    MsgpackSerde as MsgpackSerde,
    SingleWriterShmObjectStorage as SingleWriterShmObjectStorage,
    SingleWriterShmRingBuffer as SingleWriterShmRingBuffer,
)
from vllm.logger import init_logger as init_logger
from vllm.utils.cache import CacheInfo as CacheInfo, LRUCache as LRUCache
from vllm.utils.jsontree import (
    json_count_leaves as json_count_leaves,
    json_map_leaves as json_map_leaves,
    json_reduce_leaves as json_reduce_leaves,
)
from vllm.utils.mem_constants import GiB_bytes as GiB_bytes, MiB_bytes as MiB_bytes
from vllm.utils.mem_utils import format_gib as format_gib

logger: Incomplete

class MultiModalProcessorCacheItem:
    item: Incomplete
    prompt_updates: Incomplete
    def __init__(
        self,
        item: MultiModalKwargsItem,
        prompt_updates: Sequence["ResolvedPromptUpdate"],
    ) -> None: ...

class MultiModalProcessorCacheItemMetadata:
    item_size: Incomplete
    prompt_updates: Incomplete
    def __init__(
        self,
        item: MultiModalKwargsItem,
        prompt_updates: Sequence["ResolvedPromptUpdate"],
    ) -> None: ...

MultiModalCacheValue: TypeAlias = (
    MultiModalProcessorCacheItem
    | MultiModalProcessorCacheItemMetadata
    | MultiModalKwargsItems
    | MultiModalKwargsItem
    | Mapping[str, NestedTensors]
)

class MultiModalCache:
    @classmethod
    def get_leaf_size(cls, leaf: object) -> int: ...
    @classmethod
    def get_item_size(
        cls, value: MultiModalCacheValue, *, debug: bool = False
    ) -> int: ...
    @classmethod
    def get_item_complexity(cls, value: MultiModalCacheValue) -> int: ...
    @classmethod
    def get_lru_cache(
        cls, capacity_gb: float, value_type: type[_V], *, debug: bool = False
    ) -> LRUCache[str, _V]: ...

class BaseMultiModalCache(ABC, Generic[_I, _O], metaclass=abc.ABCMeta):
    @abstractmethod
    def get_and_update_item(self, mm_item: _I, mm_hash: str) -> _O: ...
    def get_and_update(
        self, mm_items: Sequence[_I], mm_hashes: list[str]
    ) -> list[_O]: ...
    @abstractmethod
    def clear_cache(self) -> None: ...

MultiModalProcessorCacheInItem: TypeAlias
MultiModalProcessorCacheOutItem: TypeAlias

class BaseMultiModalProcessorCache(
    BaseMultiModalCache[
        MultiModalProcessorCacheInItem, MultiModalProcessorCacheOutItem
    ],
    metaclass=abc.ABCMeta,
):
    @abstractmethod
    def is_cached_item(self, mm_hash: str) -> bool: ...
    def is_cached(self, mm_hashes: list[str]) -> list[bool]: ...
    def close(self) -> None: ...
    @abstractmethod
    def touch_sender_cache_item(self, mm_hash: str) -> None: ...
    @abstractmethod
    def make_stats(self, *, delta: bool = False) -> CacheInfo: ...

class MultiModalProcessorOnlyCache(BaseMultiModalProcessorCache):
    def __init__(self, model_config: ModelConfig) -> None: ...
    @override
    def is_cached_item(self, mm_hash: str) -> bool: ...
    @override
    def get_and_update_item(
        self, mm_item: MultiModalProcessorCacheInItem, mm_hash: str
    ) -> MultiModalProcessorCacheOutItem: ...
    @override
    def touch_sender_cache_item(self, mm_hash: str) -> None: ...
    @override
    def clear_cache(self) -> None: ...
    @override
    def make_stats(self, *, delta: bool = False) -> CacheInfo: ...

class MultiModalProcessorSenderCache(BaseMultiModalProcessorCache):
    def __init__(self, model_config: ModelConfig) -> None: ...
    @override
    def is_cached_item(self, mm_hash: str) -> bool: ...
    @override
    def get_and_update_item(
        self, mm_item: MultiModalProcessorCacheInItem, mm_hash: str
    ) -> MultiModalProcessorCacheOutItem: ...
    @override
    def touch_sender_cache_item(self, mm_hash: str) -> None: ...
    @override
    def clear_cache(self) -> None: ...
    @override
    def make_stats(self, *, delta: bool = False) -> CacheInfo: ...

class ShmObjectStoreSenderCache(BaseMultiModalProcessorCache):
    world_size: Incomplete
    def __init__(self, vllm_config: VllmConfig) -> None: ...
    @override
    def is_cached_item(self, mm_hash: str) -> bool: ...
    @override
    def get_and_update_item(
        self, mm_item: MultiModalProcessorCacheInItem, mm_hash: str
    ) -> MultiModalProcessorCacheOutItem: ...
    @override
    def touch_sender_cache_item(self, mm_hash: str) -> None: ...
    @override
    def clear_cache(self) -> None: ...
    @override
    def make_stats(self, *, delta: bool = False) -> CacheInfo: ...
    @override
    def close(self) -> None: ...
    def remove_dangling_items(self) -> None: ...
    def address_as_item(
        self, address: int, monotonic_id: int
    ) -> MultiModalKwargsItem: ...

class BaseMultiModalReceiverCache(
    BaseMultiModalCache[MultiModalKwargsItem | None, MultiModalKwargsItem],
    metaclass=abc.ABCMeta,
):
    def get_and_update_features(
        self, mm_features: list["MultiModalFeatureSpec"]
    ) -> list["MultiModalFeatureSpec"]: ...
    @abstractmethod
    def touch_receiver_cache_item(
        self, mm_hash: str, mm_item: MultiModalKwargsItem | None = None
    ) -> None: ...

class MultiModalReceiverCache(BaseMultiModalReceiverCache):
    def __init__(self, model_config: ModelConfig) -> None: ...
    @override
    def get_and_update_item(
        self, mm_item: MultiModalKwargsItem | None, mm_hash: str
    ) -> MultiModalKwargsItem: ...
    @override
    def touch_receiver_cache_item(
        self, mm_hash: str, mm_item: MultiModalKwargsItem | None = None
    ) -> None: ...
    @override
    def clear_cache(self) -> None: ...

class ShmObjectStoreReceiverCache(BaseMultiModalReceiverCache):
    world_size: Incomplete
    def __init__(
        self, vllm_config: VllmConfig, shared_worker_lock: LockType
    ) -> None: ...
    @override
    def get_and_update_item(
        self, mm_item: MultiModalKwargsItem | None, mm_hash: str
    ) -> MultiModalKwargsItem: ...
    @override
    def touch_receiver_cache_item(
        self, mm_hash: str, mm_item: MultiModalKwargsItem | None = None
    ) -> None: ...
    @override
    def clear_cache(self) -> None: ...
