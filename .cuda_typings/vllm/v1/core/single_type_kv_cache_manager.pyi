import abc
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from vllm.utils.math_utils import cdiv as cdiv
from vllm.v1.core.block_pool import BlockPool as BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHashList as BlockHashList,
    BlockHashWithGroupId as BlockHashWithGroupId,
    KVCacheBlock as KVCacheBlock,
)
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec as ChunkedLocalAttentionSpec,
    CrossAttentionSpec as CrossAttentionSpec,
    FullAttentionSpec as FullAttentionSpec,
    KVCacheSpec as KVCacheSpec,
    MLAAttentionSpec as MLAAttentionSpec,
    MambaSpec as MambaSpec,
    SinkFullAttentionSpec as SinkFullAttentionSpec,
    SlidingWindowSpec as SlidingWindowSpec,
)
from vllm.v1.request import Request as Request

class SingleTypeKVCacheManager(ABC, metaclass=abc.ABCMeta):
    block_size: Incomplete
    dcp_world_size: Incomplete
    pcp_world_size: Incomplete
    kv_cache_spec: Incomplete
    block_pool: Incomplete
    enable_caching: Incomplete
    new_block_ids: list[int]
    req_to_blocks: defaultdict[str, list[KVCacheBlock]]
    num_cached_block: dict[str, int]
    kv_cache_group_id: Incomplete
    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        block_pool: BlockPool,
        enable_caching: bool,
        kv_cache_group_id: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> None: ...
    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_tokens_main_model: int,
    ) -> int: ...
    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None: ...
    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]: ...
    def take_new_block_ids(self) -> list[int]: ...
    def cache_blocks(self, request: Request, num_tokens: int) -> None: ...
    def free(self, request_id: str) -> None: ...
    @abstractmethod
    def get_num_common_prefix_blocks(self, running_request_id: str) -> int: ...
    @classmethod
    @abstractmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]: ...
    def remove_skipped_blocks(
        self, request_id: str, total_computed_tokens: int
    ) -> None: ...
    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int: ...
    def new_step_starts(self) -> None: ...

class FullAttentionManager(SingleTypeKVCacheManager):
    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]: ...
    def get_num_common_prefix_blocks(self, running_request_id: str) -> int: ...

class SlidingWindowManager(SingleTypeKVCacheManager):
    sliding_window: Incomplete
    def __init__(self, kv_cache_spec: SlidingWindowSpec, **kwargs) -> None: ...
    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]: ...
    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int: ...
    def get_num_common_prefix_blocks(self, running_request_id: str) -> int: ...

class ChunkedLocalAttentionManager(SingleTypeKVCacheManager):
    attention_chunk_size: Incomplete
    def __init__(self, kv_cache_spec: ChunkedLocalAttentionSpec, **kwargs) -> None: ...
    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]: ...
    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int: ...
    def get_num_common_prefix_blocks(self, running_request_id: str) -> int: ...

class MambaManager(SingleTypeKVCacheManager):
    cached_blocks_this_step: set[BlockHashWithGroupId]
    mamba_cache_mode: Incomplete
    num_speculative_blocks: int
    last_state_block_idx: dict[str, int]
    def __init__(
        self, kv_cache_spec: MambaSpec, block_pool: BlockPool, **kwargs
    ) -> None: ...
    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]: ...
    def remove_skipped_blocks(
        self, request_id: str, num_computed_tokens: int
    ) -> None: ...
    def get_num_common_prefix_blocks(self, running_request_id: str) -> int: ...
    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_tokens_main_model: int,
    ) -> int: ...
    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]: ...
    def free(self, request_id: str) -> None: ...
    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int: ...
    def cache_blocks(self, request: Request, num_tokens: int) -> None: ...
    def new_step_starts(self) -> None: ...

class CrossAttentionManager(SingleTypeKVCacheManager):
    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None: ...
    def cache_blocks(self, request: Request, num_tokens: int) -> None: ...
    def get_num_common_prefix_blocks(self, running_request_id: str) -> int: ...
    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]: ...

class SinkFullAttentionManager(FullAttentionManager):
    sink_blocks: Incomplete
    def __init__(
        self,
        kv_cache_spec: SinkFullAttentionSpec,
        block_pool: BlockPool,
        enable_caching: bool,
        kv_cache_group_id: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> None: ...

spec_manager_map: dict[type[KVCacheSpec], type[SingleTypeKVCacheManager]]

def get_manager_for_kv_cache_spec(
    kv_cache_spec: KVCacheSpec, **kwargs
) -> SingleTypeKVCacheManager: ...
