class KVCacheBlock:
    def __init__(self, block_id: int) -> None: ...

class FreeKVCacheBlockQueue:
    def append_n(self, blocks: list[KVCacheBlock]) -> None: ...

class BlockPool:
    blocks: list[KVCacheBlock]
    free_block_queue: FreeKVCacheBlockQueue
    num_gpu_blocks: int
