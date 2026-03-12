import abc
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from vllm.v1.core.kv_cache_utils import BlockHash as BlockHash

class LoadStoreSpec(ABC, metaclass=abc.ABCMeta):
    @staticmethod
    @abstractmethod
    def medium() -> str: ...

@dataclass
class PrepareStoreOutput:
    block_hashes_to_store: list[BlockHash]
    store_spec: LoadStoreSpec
    block_hashes_evicted: list[BlockHash]

@dataclass
class OffloadingEvent:
    block_hashes: list[BlockHash]
    block_size: int
    medium: str
    removed: bool

class OffloadingManager(ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    def lookup(self, block_hashes: Iterable[BlockHash]) -> int | None: ...
    @abstractmethod
    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec: ...
    def touch(self, block_hashes: Iterable[BlockHash]): ...
    def complete_load(self, block_hashes: Iterable[BlockHash]): ...
    @abstractmethod
    def prepare_store(
        self, block_hashes: Iterable[BlockHash]
    ) -> PrepareStoreOutput | None: ...
    def complete_store(
        self, block_hashes: Iterable[BlockHash], success: bool = True
    ): ...
    def take_events(self) -> Iterable[OffloadingEvent]: ...
