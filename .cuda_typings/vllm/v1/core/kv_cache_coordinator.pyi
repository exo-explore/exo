import abc
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Sequence
from vllm.v1.core.block_pool import BlockPool as BlockPool
from vllm.v1.core.kv_cache_metrics import (
    KVCacheMetricsCollector as KVCacheMetricsCollector,
)
from vllm.v1.core.kv_cache_utils import (
    BlockHash as BlockHash,
    BlockHashList as BlockHashList,
    BlockHashListWithBlockSize as BlockHashListWithBlockSize,
    KVCacheBlock as KVCacheBlock,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    CrossAttentionManager as CrossAttentionManager,
    SingleTypeKVCacheManager as SingleTypeKVCacheManager,
    get_manager_for_kv_cache_spec as get_manager_for_kv_cache_spec,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec as FullAttentionSpec,
    KVCacheConfig as KVCacheConfig,
    KVCacheSpec as KVCacheSpec,
)
from vllm.v1.request import Request as Request

class KVCacheCoordinator(ABC, metaclass=abc.ABCMeta):
    kv_cache_config: Incomplete
    max_model_len: Incomplete
    enable_caching: Incomplete
    block_pool: Incomplete
    use_eagle: Incomplete
    single_type_managers: Incomplete
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ) -> None: ...
    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_encoder_tokens: int,
        total_computed_tokens: int,
        num_tokens_main_model: int,
    ) -> int: ...
    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None: ...
    def allocate_new_blocks(
        self,
        request_id: str,
        num_tokens: int,
        num_tokens_main_model: int,
        num_encoder_tokens: int = 0,
    ) -> tuple[list[KVCacheBlock], ...]: ...
    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None: ...
    def free(self, request_id: str) -> None: ...
    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]: ...
    def remove_skipped_blocks(
        self, request_id: str, total_computed_tokens: int
    ) -> None: ...
    def get_blocks(self, request_id: str) -> tuple[list[KVCacheBlock], ...]: ...
    @abstractmethod
    def find_longest_cache_hit(
        self, block_hashes: list[BlockHash], max_cache_hit_length: int
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]: ...
    def new_step_starts(self) -> None: ...

class KVCacheCoordinatorNoPrefixCache(KVCacheCoordinator):
    num_single_type_manager: Incomplete
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        use_eagle: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ) -> None: ...
    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]: ...
    def find_longest_cache_hit(
        self, block_hashes: list[BlockHash], max_cache_hit_length: int
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]: ...

class UnitaryKVCacheCoordinator(KVCacheCoordinator):
    kv_cache_spec: Incomplete
    block_size: Incomplete
    dcp_world_size: Incomplete
    pcp_world_size: Incomplete
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ) -> None: ...
    def find_longest_cache_hit(
        self, block_hashes: list[BlockHash], max_cache_hit_length: int
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]: ...

class HybridKVCacheCoordinator(KVCacheCoordinator):
    hash_block_size: Incomplete
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ) -> None: ...
    attention_groups: Incomplete
    lcm_block_size: Incomplete
    def verify_and_split_kv_cache_groups(self) -> None: ...
    def find_longest_cache_hit(
        self, block_hashes: list[BlockHash], max_cache_hit_length: int
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]: ...

def get_kv_cache_coordinator(
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    use_eagle: bool,
    enable_caching: bool,
    enable_kv_cache_events: bool,
    dcp_world_size: int,
    pcp_world_size: int,
    hash_block_size: int,
    metrics_collector: KVCacheMetricsCollector | None = None,
) -> KVCacheCoordinator: ...
