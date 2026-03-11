from vllm.v1.core.kv_cache_utils import BlockPool

class KVCacheManager:
    block_pool: BlockPool
    def __init__(self, *args: object, **kwargs: object) -> None: ...
    def allocate_slots(
        self, request: object, num_new_tokens: int, *args: object, **kwargs: object
    ) -> object | None: ...
