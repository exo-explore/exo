from collections.abc import Sequence

from pydantic import Field, model_validator

from exo.utils.pydantic_ext import FrozenModel

DEFAULT_SINK_TOKENS = 4
DEFAULT_RECENT_WINDOW_TOKENS = 256
DEFAULT_HEAVY_HITTER_TOKENS = 256
DEFAULT_MAX_CACHE_TOKENS = 1024


class StreamingCacheConfig(FrozenModel):
    """Configuration for O(1) constant-memory StreamingLLM + H2O KV cache eviction."""

    max_cache_tokens: int = DEFAULT_MAX_CACHE_TOKENS
    num_sink_tokens: int = DEFAULT_SINK_TOKENS
    recent_window_tokens: int = DEFAULT_RECENT_WINDOW_TOKENS
    heavy_hitter_tokens: int = DEFAULT_HEAVY_HITTER_TOKENS

    @model_validator(mode="after")
    def validate_budget_fits(self) -> "StreamingCacheConfig":
        required_minimum = (
            self.num_sink_tokens + self.recent_window_tokens + self.heavy_hitter_tokens
        )
        if self.max_cache_tokens < required_minimum:
            raise ValueError(
                f"max_cache_tokens ({self.max_cache_tokens}) must be >= "
                f"sinks ({self.num_sink_tokens}) + recent ({self.recent_window_tokens}) + "
                f"heavy hitters ({self.heavy_hitter_tokens}) = {required_minimum}"
            )
        return self


class EvictionMetrics(FrozenModel):
    """Telemetry metrics tracking cache eviction efficiency and memory savings."""

    total_tokens_seen: int
    current_cache_size: int
    evicted_tokens_count: int

    @property
    def memory_saving_percentage(self) -> float:
        if self.total_tokens_seen == 0:
            return 0.0
        saved = max(0, self.total_tokens_seen - self.current_cache_size)
        return (saved / self.total_tokens_seen) * 100.0


class EvictionResult(FrozenModel):
    """Result of an eviction step indicating retained and evicted positions."""

    retained_indices: Sequence[int] = Field(default_factory=tuple)
    evicted_indices: Sequence[int] = Field(default_factory=tuple)
    active_tokens_count: int = 0
