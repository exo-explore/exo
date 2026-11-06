from typing import Any, Literal

from pydantic import ConfigDict, Field

from exo.utils.pydantic_ext import FrozenModel

StorageTier = Literal["vram", "host_ram", "nvme_ssd"]
DEFAULT_LOOKAHEAD_LAYERS: int = 4
DEFAULT_HOT_EXPERTS_PER_LAYER: int = 2


class OffloadedExpertWeight(FrozenModel):
    """Represents an MoE expert weight matrix across tiered storage (VRAM/RAM/SSD)."""

    model_config = ConfigDict(frozen=True, strict=True)

    expert_id: int
    layer_id: int
    num_params: int = 0
    size_in_bytes: int = 0
    current_tier: StorageTier = "nvme_ssd"
    is_pinned_hot: bool = False
    access_count: int = 0
    last_accessed_timestamp: float = 0.0
    weights_data: tuple[float, ...] = Field(default_factory=tuple)

    @property
    def expert_key(self) -> str:
        return f"l{self.layer_id}_e{self.expert_id}"


class ExpertCacheSlot(FrozenModel):
    """A physical memory slot in the fast VRAM/RAM pool."""

    model_config = ConfigDict(frozen=True, strict=True)

    slot_id: int
    assigned_expert_key: str | None = None
    is_hot_slot: bool = False
    last_used_timestamp: float = 0.0


class MoEOffloadTelemetry(FrozenModel):
    """Telemetry capturing MoE hierarchical swapping and speculative prefetch masking."""

    model_config = ConfigDict(frozen=True, strict=True)

    total_expert_requests: int = 0
    hot_cache_hits: int = 0
    speculative_prefetch_hits: int = 0
    cold_ssd_misses: int = 0
    total_swap_in_bytes: int = 0
    io_time_masked_ms: float = 0.0
    total_io_latency_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        if self.total_expert_requests == 0:
            return 0.0
        return (self.hot_cache_hits + self.speculative_prefetch_hits) / self.total_expert_requests

    @property
    def prefetch_masking_ratio(self) -> float:
        if self.total_io_latency_ms <= 1e-6:
            return 1.0 if self.speculative_prefetch_hits > 0 else 0.0
        return min(1.0, self.io_time_masked_ms / self.total_io_latency_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_expert_requests": self.total_expert_requests,
            "hot_cache_hits": self.hot_cache_hits,
            "speculative_prefetch_hits": self.speculative_prefetch_hits,
            "cold_ssd_misses": self.cold_ssd_misses,
            "hit_rate_pct": round(self.hit_rate * 100.0, 2),
            "total_swap_in_bytes": self.total_swap_in_bytes,
            "io_time_masked_ms": round(self.io_time_masked_ms, 3),
            "total_io_latency_ms": round(self.total_io_latency_ms, 3),
            "masking_efficiency_pct": round(self.prefetch_masking_ratio * 100.0, 2),
        }
