from collections.abc import Sequence

from vllm.v1.core.kv_cache_utils import BlockPool, KVCacheBlock
from vllm.v1.kv_cache_interface import KVCacheConfig

class KVCacheBlocks:
    blocks: tuple[Sequence[KVCacheBlock], ...]
    def __init__(self, blocks: tuple[Sequence[KVCacheBlock], ...]) -> None: ...
    def get_block_ids(self) -> tuple[list[int], ...]: ...

class KVCacheManager:
    block_pool: BlockPool
    kv_cache_config: KVCacheConfig
    enable_caching: bool
    num_kv_cache_groups: int
    coordinator: object
    def __init__(self, *args: object, **kwargs: object) -> None: ...
    def allocate_slots(
        self, request: object, num_new_tokens: int, *args: object, **kwargs: object
    ) -> KVCacheBlocks | None: ...
    def get_computed_blocks(self, request: object) -> tuple[KVCacheBlocks, int]: ...
    def create_kv_cache_blocks(self, blocks: tuple[list[KVCacheBlock], ...]) -> KVCacheBlocks: ...
