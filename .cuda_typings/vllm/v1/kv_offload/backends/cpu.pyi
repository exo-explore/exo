from _typeshed import Incomplete
from collections.abc import Iterable
from vllm.v1.core.kv_cache_utils import BlockHash as BlockHash
from vllm.v1.kv_offload.abstract import LoadStoreSpec as LoadStoreSpec
from vllm.v1.kv_offload.backend import Backend as Backend, BlockStatus as BlockStatus
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec as CPULoadStoreSpec

class CPUBlockStatus(BlockStatus):
    block_id: Incomplete
    def __init__(self, block_id: int) -> None: ...

class CPUBackend(Backend):
    num_blocks: int
    num_allocated_blocks: int
    allocated_blocks_free_list: list[int]
    def __init__(self, block_size: int, num_blocks: int) -> None: ...
    def get_num_free_blocks(self): ...
    def allocate_blocks(self, block_hashes: list[BlockHash]) -> list[BlockStatus]: ...
    def free(self, block: BlockStatus): ...
    def get_load_store_spec(
        self, block_hashes: Iterable[BlockHash], blocks: Iterable[BlockStatus]
    ) -> LoadStoreSpec: ...
