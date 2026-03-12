from _typeshed import Incomplete
from collections import OrderedDict
from collections.abc import Iterable
from vllm.v1.core.kv_cache_utils import BlockHash as BlockHash
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec as LoadStoreSpec,
    OffloadingEvent as OffloadingEvent,
    OffloadingManager as OffloadingManager,
    PrepareStoreOutput as PrepareStoreOutput,
)

class FilterReusedOffloadingManager(OffloadingManager):
    store_threshold: Incomplete
    max_tracker_size: Incomplete
    counts: OrderedDict[BlockHash, int]
    def __init__(
        self,
        backing: OffloadingManager,
        store_threshold: int = 2,
        max_tracker_size: int = 64000,
    ) -> None: ...
    def lookup(self, block_hashes: Iterable[BlockHash]) -> int | None: ...
    def prepare_store(
        self, block_hashes: Iterable[BlockHash]
    ) -> PrepareStoreOutput | None: ...
    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec: ...
    def touch(self, block_hashes: Iterable[BlockHash]) -> None: ...
    def complete_load(self, block_hashes: Iterable[BlockHash]) -> None: ...
    def complete_store(
        self, block_hashes: Iterable[BlockHash], success: bool = True
    ) -> None: ...
    def take_events(self) -> Iterable[OffloadingEvent]: ...
