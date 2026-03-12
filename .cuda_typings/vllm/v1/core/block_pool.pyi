from _typeshed import Incomplete
from collections.abc import Iterable, Sequence
from vllm.distributed.kv_events import (
    AllBlocksCleared as AllBlocksCleared,
    BlockRemoved as BlockRemoved,
    BlockStored as BlockStored,
    KVCacheEvent as KVCacheEvent,
    MEDIUM_GPU as MEDIUM_GPU,
)
from vllm.logger import init_logger as init_logger
from vllm.v1.core.kv_cache_metrics import (
    KVCacheMetricsCollector as KVCacheMetricsCollector,
)
from vllm.v1.core.kv_cache_utils import (
    BlockHash as BlockHash,
    BlockHashList as BlockHashList,
    BlockHashListWithBlockSize as BlockHashListWithBlockSize,
    BlockHashWithGroupId as BlockHashWithGroupId,
    ExternalBlockHash as ExternalBlockHash,
    FreeKVCacheBlockQueue as FreeKVCacheBlockQueue,
    KVCacheBlock as KVCacheBlock,
    generate_block_hash_extra_keys as generate_block_hash_extra_keys,
    get_block_hash as get_block_hash,
    make_block_hash_with_group_id as make_block_hash_with_group_id,
    maybe_convert_block_hash as maybe_convert_block_hash,
)
from vllm.v1.request import Request as Request

logger: Incomplete

class BlockHashToBlockMap:
    def __init__(self) -> None: ...
    def get_one_block(self, key: BlockHashWithGroupId) -> KVCacheBlock | None: ...
    def insert(self, key: BlockHashWithGroupId, block: KVCacheBlock) -> None: ...
    def pop(self, key: BlockHashWithGroupId, block_id: int) -> KVCacheBlock | None: ...
    def __len__(self) -> int: ...

class BlockPool:
    num_gpu_blocks: Incomplete
    enable_caching: Incomplete
    hash_block_size: Incomplete
    blocks: list[KVCacheBlock]
    free_block_queue: Incomplete
    cached_block_hash_to_block: BlockHashToBlockMap
    null_block: Incomplete
    enable_kv_cache_events: Incomplete
    kv_event_queue: list[KVCacheEvent]
    metrics_collector: Incomplete
    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        hash_block_size: int,
        enable_kv_cache_events: bool = False,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ) -> None: ...
    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int]
    ) -> list[KVCacheBlock] | None: ...
    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
    ) -> None: ...
    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]: ...
    def touch(self, blocks: Sequence[KVCacheBlock]) -> None: ...
    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None: ...
    def evict_blocks(self, block_ids: set[int]) -> None: ...
    def reset_prefix_cache(self) -> bool: ...
    def get_num_free_blocks(self) -> int: ...
    def get_usage(self) -> float: ...
    def take_events(self) -> list[KVCacheEvent]: ...
