from typing import Any

from pydantic import ConfigDict, Field

from exo.utils.pydantic_ext import FrozenModel

DEFAULT_BLOCK_SIZE: int = 16
DEFAULT_LOOKAHEAD_BLOCKS: int = 2
DEFAULT_SINK_BLOCKS: int = 1


class PagedKVBlock(FrozenModel):
    """Represents a discrete, fixed-size memory page holding Key-Value caches."""

    model_config = ConfigDict(frozen=True, strict=True)

    block_id: str
    sequence_id: str
    logical_block_index: int
    block_size: int = DEFAULT_BLOCK_SIZE
    num_heads: int = 32
    head_dim: int = 128
    device_location: str = "local_ram"  # e.g. "local_vram", "local_ram", "nvme_ssd", "remote_node"
    is_pinned: bool = False
    access_count: int = 0
    last_accessed_timestamp: float = 0.0
    keys_data: tuple[tuple[float, ...], ...] = Field(default_factory=tuple)
    values_data: tuple[tuple[float, ...], ...] = Field(default_factory=tuple)

    @property
    def capacity_tokens(self) -> int:
        return self.block_size

    @property
    def current_tokens_count(self) -> int:
        return len(self.keys_data)

    @property
    def is_full(self) -> bool:
        return self.current_tokens_count >= self.capacity_tokens

    @property
    def size_in_bytes(self) -> int:
        """Estimate payload size in bytes assuming float32 representation."""
        floats_count = self.current_tokens_count * self.num_heads * self.head_dim * 2
        return floats_count * 4


class BlockTable(FrozenModel):
    """Maps logical token sequence blocks to physical or remote PagedKVBlocks."""

    model_config = ConfigDict(frozen=True, strict=True)

    sequence_id: str
    block_size: int = DEFAULT_BLOCK_SIZE
    logical_to_physical: dict[int, str] = Field(default_factory=dict)

    def get_block_id(self, logical_index: int) -> str | None:
        return self.logical_to_physical.get(logical_index)

    def total_mapped_blocks(self) -> int:
        return len(self.logical_to_physical)

    def total_capacity_tokens(self) -> int:
        return self.total_mapped_blocks() * self.block_size


class PrefetchRequest(FrozenModel):
    """Represents an asynchronous speculative request to preload candidate KV blocks."""

    model_config = ConfigDict(frozen=True, strict=True)

    request_id: str
    sequence_id: str
    target_logical_block_indices: tuple[int, ...]
    source_location: str = "nvme_ssd"
    target_location: str = "local_ram"
    urgency_priority: int = 1  # 1 = speculative future, 2 = immediate next step


class PrefetchTelemetry(FrozenModel):
    """Telemetry capturing speculative prefetch hit rates and I/O latency masking."""

    model_config = ConfigDict(frozen=True, strict=True)

    total_prefetches: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    io_time_masked_ms: float = 0.0
    total_io_latency_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total

    @property
    def masking_efficiency(self) -> float:
        if self.total_io_latency_ms <= 1e-6:
            return 1.0 if self.cache_hits > 0 else 0.0
        return min(1.0, self.io_time_masked_ms / self.total_io_latency_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_prefetches": self.total_prefetches,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_pct": round(self.hit_rate * 100.0, 2),
            "io_time_masked_ms": round(self.io_time_masked_ms, 3),
            "total_io_latency_ms": round(self.total_io_latency_ms, 3),
            "masking_efficiency_pct": round(self.masking_efficiency * 100.0, 2),
        }
