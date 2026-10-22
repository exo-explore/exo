
import numpy as np
from numpy.typing import NDArray

from exo.shared.types.streaming_cache import (
    EvictionMetrics,
    EvictionResult,
    StreamingCacheConfig,
)
from exo.worker.cache.attention_sinks import AttentionSinkManager
from exo.worker.cache.heavy_hitter import HeavyHitterEvictionEngine


class StreamingKVCacheManager:
    """Unified constant-memory KV cache manager integrating StreamingLLM and H2O eviction."""

    def __init__(self, config: StreamingCacheConfig | None = None) -> None:
        self._config = config or StreamingCacheConfig()
        self._sink_manager = AttentionSinkManager(self._config.num_sink_tokens)
        self._eviction_engine = HeavyHitterEvictionEngine(self._config)

        self._cached_key: NDArray[np.floating] | None = None
        self._cached_value: NDArray[np.floating] | None = None
        self._total_tokens_seen: int = 0
        self._total_evicted_tokens: int = 0

    @property
    def config(self) -> StreamingCacheConfig:
        return self._config

    @property
    def current_cache_length(self) -> int:
        return self._eviction_engine.current_cached_tokens_count

    def reset(self) -> None:
        """Resets all KV caches and tracking states."""
        self._cached_key = None
        self._cached_value = None
        self._eviction_engine.reset()
        self._total_tokens_seen = 0
        self._total_evicted_tokens = 0

    def append_and_evict(
        self,
        new_key: NDArray[np.floating],
        new_value: NDArray[np.floating],
        attention_weights: NDArray[np.floating] | None = None,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], EvictionResult]:
        """Appends incoming token KV tensors along sequence axis, evicting if budget exceeded.

        Assumes sequence dimension is axis -2 (standard Transformer format: (..., seq_len, head_dim)).
        """
        k_arr = np.asarray(new_key)
        v_arr = np.asarray(new_value)

        new_tokens_count = k_arr.shape[-2]
        self._total_tokens_seen += new_tokens_count

        # Concatenate with existing cache along sequence dimension (axis -2)
        if self._cached_key is None or self._cached_value is None:
            combined_k = k_arr
            combined_v = v_arr
        else:
            combined_k = np.concatenate([self._cached_key, k_arr], axis=-2)
            combined_v = np.concatenate([self._cached_value, v_arr], axis=-2)

        # Update scores in eviction engine
        self._eviction_engine.update_scores_with_new_tokens(
            new_token_count=new_tokens_count,
            attention_weights=attention_weights,
        )

        # Compute eviction if budget is exceeded
        eviction_result = self._eviction_engine.compute_eviction()

        if len(eviction_result.evicted_indices) > 0:
            retained = list(eviction_result.retained_indices)
            # Slice along sequence dimension (axis -2)
            combined_k = np.take(combined_k, retained, axis=-2)
            combined_v = np.take(combined_v, retained, axis=-2)
            self._total_evicted_tokens += len(eviction_result.evicted_indices)

        self._cached_key = combined_k
        self._cached_value = combined_v

        return self._cached_key, self._cached_value, eviction_result

    def get_active_kv(self) -> tuple[NDArray[np.floating] | None, NDArray[np.floating] | None]:
        """Returns the current active in-memory KV cache tensors."""
        return self._cached_key, self._cached_value

    def get_metrics(self) -> EvictionMetrics:
        """Returns telemetry metrics regarding cache utilization and savings."""
        return EvictionMetrics(
            total_tokens_seen=self._total_tokens_seen,
            current_cache_size=self.current_cache_length,
            evicted_tokens_count=self._total_evicted_tokens,
        )
