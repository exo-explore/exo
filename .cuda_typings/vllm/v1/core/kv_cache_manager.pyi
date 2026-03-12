from _typeshed import Incomplete
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, overload
from vllm.distributed.kv_events import KVCacheEvent as KVCacheEvent
from vllm.logger import init_logger as init_logger
from vllm.v1.core.kv_cache_coordinator import (
    get_kv_cache_coordinator as get_kv_cache_coordinator,
)
from vllm.v1.core.kv_cache_metrics import (
    KVCacheMetricsCollector as KVCacheMetricsCollector,
)
from vllm.v1.core.kv_cache_utils import KVCacheBlock as KVCacheBlock
from vllm.v1.kv_cache_interface import KVCacheConfig as KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats as PrefixCacheStats
from vllm.v1.request import Request as Request

logger: Incomplete

@dataclass
class KVCacheBlocks:
    blocks: tuple[Sequence[KVCacheBlock], ...]
    def __add__(self, other: KVCacheBlocks) -> KVCacheBlocks: ...
    @overload
    def get_block_ids(
        self, allow_none: Literal[False] = False
    ) -> tuple[list[int], ...]: ...
    @overload
    def get_block_ids(
        self, allow_none: Literal[True] = True
    ) -> tuple[list[int], ...] | None: ...
    def get_unhashed_block_ids(self) -> list[int]: ...
    def get_unhashed_block_ids_all_groups(self) -> list[list[int]]: ...
    def new_empty(self) -> KVCacheBlocks: ...

class KVCacheManager:
    max_model_len: Incomplete
    enable_caching: Incomplete
    use_eagle: Incomplete
    log_stats: Incomplete
    metrics_collector: Incomplete
    prefix_cache_stats: Incomplete
    coordinator: Incomplete
    num_kv_cache_groups: Incomplete
    block_pool: Incomplete
    kv_cache_config: Incomplete
    empty_kv_cache_blocks: Incomplete
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        hash_block_size: int,
        enable_caching: bool = True,
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ) -> None: ...
    @property
    def usage(self) -> float: ...
    def make_prefix_cache_stats(self) -> PrefixCacheStats | None: ...
    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]: ...
    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
    ) -> KVCacheBlocks | None: ...
    def free(self, request: Request) -> None: ...
    def remove_skipped_blocks(
        self, request_id: str, total_computed_tokens: int
    ) -> None: ...
    def evict_blocks(self, block_ids: set[int]) -> None: ...
    def reset_prefix_cache(self) -> bool: ...
    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]: ...
    def take_events(self) -> list[KVCacheEvent]: ...
    def get_blocks(self, request_id: str) -> KVCacheBlocks: ...
    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]: ...
    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None: ...
    def create_kv_cache_blocks(
        self, blocks: tuple[list[KVCacheBlock], ...]
    ) -> KVCacheBlocks: ...
    def take_new_block_ids(self) -> list[int]: ...
    def new_step_starts(self) -> None: ...
