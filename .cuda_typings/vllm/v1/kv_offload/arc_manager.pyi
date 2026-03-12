from collections import OrderedDict
from collections.abc import Iterable
from vllm.v1.core.kv_cache_utils import BlockHash as BlockHash
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec as LoadStoreSpec,
    OffloadingEvent as OffloadingEvent,
    OffloadingManager as OffloadingManager,
    PrepareStoreOutput as PrepareStoreOutput,
)
from vllm.v1.kv_offload.backend import Backend as Backend, BlockStatus as BlockStatus

class ARCOffloadingManager(OffloadingManager):
    backend: Backend
    target_t1_size: float
    t1: OrderedDict[BlockHash, BlockStatus]
    t2: OrderedDict[BlockHash, BlockStatus]
    b1: OrderedDict[BlockHash, None]
    b2: OrderedDict[BlockHash, None]
    events: list[OffloadingEvent] | None
    cache_capacity: int
    def __init__(self, backend: Backend, enable_events: bool = False) -> None: ...
    def lookup(self, block_hashes: Iterable[BlockHash]) -> int | None: ...
    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec: ...
    def touch(self, block_hashes: Iterable[BlockHash]): ...
    def complete_load(self, block_hashes: Iterable[BlockHash]): ...
    def prepare_store(
        self, block_hashes: Iterable[BlockHash]
    ) -> PrepareStoreOutput | None: ...
    def complete_store(
        self, block_hashes: Iterable[BlockHash], success: bool = True
    ): ...
    def take_events(self) -> Iterable[OffloadingEvent]: ...
