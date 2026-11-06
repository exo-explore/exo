import numpy as np
from numpy.typing import NDArray

from exo.shared.types.streaming_cache import (
    EvictionResult,
    StreamingCacheConfig,
)


class HeavyHitterEvictionEngine:
    """H2O (Heavy Hitter Oracle) dynamic token eviction engine.

    Accumulates cross-head attention scores to identify high-value tokens,
    evicting low-utility tokens to guarantee O(1) constant cache footprint.
    """

    def __init__(self, config: StreamingCacheConfig) -> None:
        self._config = config
        self._accumulated_scores: NDArray[np.float32] = np.empty(0, dtype=np.float32)

    @property
    def config(self) -> StreamingCacheConfig:
        return self._config

    @property
    def current_cached_tokens_count(self) -> int:
        return len(self._accumulated_scores)

    def reset(self) -> None:
        """Resets scores buffer when starting a new sequence."""
        self._accumulated_scores = np.empty(0, dtype=np.float32)

    def update_scores_with_new_tokens(
        self,
        new_token_count: int,
        attention_weights: NDArray[np.floating] | None = None,
    ) -> None:
        """Appends new token score entries and accumulates newly observed attention weights."""
        if new_token_count <= 0:
            return

        # Initialize score entries for newly appended tokens (default base priority 1.0)
        new_scores = np.ones(new_token_count, dtype=np.float32)
        self._accumulated_scores = np.concatenate([self._accumulated_scores, new_scores])

        # If step attention weights are provided, accumulate across heads
        if attention_weights is not None:
            weights = np.asarray(attention_weights, dtype=np.float32)
            # Sum across batch and head dimensions to get 1D scores per token
            if weights.ndim > 1:
                # Collapse all leading dimensions down to the sequence length
                collapsed = np.sum(weights, axis=tuple(range(weights.ndim - 1)))
            else:
                collapsed = weights

            # Match lengths
            min_len = min(len(self._accumulated_scores), len(collapsed))
            self._accumulated_scores[:min_len] += collapsed[:min_len]

    def compute_eviction(self) -> EvictionResult:
        """Computes retained vs evicted token indices based on Sinks + H2O + Recent Window."""
        total_tokens = len(self._accumulated_scores)
        if total_tokens <= self._config.max_cache_tokens:
            # Under budget: retain all
            return EvictionResult(
                retained_indices=tuple(range(total_tokens)),
                evicted_indices=(),
                active_tokens_count=total_tokens,
            )

        num_sinks = min(self._config.num_sink_tokens, total_tokens)
        recent_window = min(self._config.recent_window_tokens, total_tokens - num_sinks)
        recent_start_index = total_tokens - recent_window

        # 1. Protected Attention Sinks
        sink_indices = set(range(num_sinks))

        # 2. Protected Recent Sliding Window
        recent_indices = set(range(recent_start_index, total_tokens))

        # 3. Middle Candidate Pool for Heavy Hitter selection
        middle_candidates = [
            i for i in range(num_sinks, recent_start_index)
            if i not in sink_indices and i not in recent_indices
        ]

        # Select Top-H heavy hitters based on accumulated attention scores
        budget_for_heavy_hitters = max(
            0,
            self._config.max_cache_tokens - num_sinks - recent_window
        )

        if len(middle_candidates) > budget_for_heavy_hitters:
            candidate_scores = self._accumulated_scores[middle_candidates]
            # Partition top candidates
            partition_idx = len(candidate_scores) - budget_for_heavy_hitters
            top_local_indices = np.argpartition(candidate_scores, partition_idx)[partition_idx:]
            retained_heavy_hitters = {middle_candidates[idx] for idx in top_local_indices}
        else:
            retained_heavy_hitters = set(middle_candidates)

        retained_set = sink_indices | recent_indices | retained_heavy_hitters
        retained_sorted = sorted(retained_set)
        evicted_sorted = [i for i in range(total_tokens) if i not in retained_set]

        # Update internal scores to match retained tokens
        self._accumulated_scores = self._accumulated_scores[retained_sorted]

        return EvictionResult(
            retained_indices=tuple(retained_sorted),
            evicted_indices=tuple(evicted_sorted),
            active_tokens_count=len(retained_sorted),
        )
