import abc
import ctypes
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Iterable
from vllm.v1.core.kv_cache_utils import BlockHash as BlockHash
from vllm.v1.kv_offload.abstract import LoadStoreSpec as LoadStoreSpec

class BlockStatus(ctypes.Structure):
    ref_cnt: int
    def __init__(self) -> None: ...
    @property
    def is_ready(self) -> bool: ...

class Backend(ABC, metaclass=abc.ABCMeta):
    block_size: Incomplete
    medium: Incomplete
    def __init__(self, block_size: int, medium: str) -> None: ...
    @abstractmethod
    def get_num_free_blocks(self): ...
    @abstractmethod
    def allocate_blocks(self, block_hashes: list[BlockHash]) -> list[BlockStatus]: ...
    @abstractmethod
    def free(self, block: BlockStatus): ...
    def get_load_store_spec(
        self, block_hashes: Iterable[BlockHash], blocks: Iterable[BlockStatus]
    ) -> LoadStoreSpec: ...
