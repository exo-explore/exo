import numpy as np
import pytest

from exo.shared.types.streaming_cache import (
    StreamingCacheConfig,
)
from exo.worker.cache.attention_sinks import AttentionSinkManager
from exo.worker.cache.heavy_hitter import HeavyHitterEvictionEngine
from exo.worker.cache.streaming_manager import StreamingKVCacheManager


def test_streaming_cache_config_validation() -> None:
    # Valid configuration
    valid_cfg = StreamingCacheConfig(
        max_cache_tokens=512,
        num_sink_tokens=4,
        recent_window_tokens=128,
        heavy_hitter_tokens=128,
    )
    assert valid_cfg.max_cache_tokens == 512

    # Invalid: components exceed max budget
    with pytest.raises(ValueError, match="must be >="):
        StreamingCacheConfig(
            max_cache_tokens=100,
            num_sink_tokens=4,
            recent_window_tokens=60,
            heavy_hitter_tokens=50,
        )


def test_attention_sink_preservation() -> None:
    sink_mgr = AttentionSinkManager(num_sink_tokens=4)
    assert sink_mgr.is_sink_index(0) is True
    assert sink_mgr.is_sink_index(3) is True
    assert sink_mgr.is_sink_index(4) is False
    assert sink_mgr.is_sink_index(100) is False

    active_indices = np.array([0, 1, 2, 3, 45, 46, 47], dtype=np.int32)
    aligned_positions = sink_mgr.generate_aligned_position_ids(active_indices, 100)
    assert aligned_positions[0] == 0
    assert aligned_positions[3] == 3


def test_h2o_heavy_hitter_eviction_mechanics() -> None:
    config = StreamingCacheConfig(
        max_cache_tokens=10,
        num_sink_tokens=2,  # Indices 0, 1
        recent_window_tokens=3,  # Last 3 indices
        heavy_hitter_tokens=2,  # 2 middle tokens to keep
    )
    engine = HeavyHitterEvictionEngine(config)

    # 1. Initialize 12 tokens
    engine.update_scores_with_new_tokens(12)

    # 2. Assign high attention weights to token 4 and token 6 (Heavy Hitters)
    custom_attention = np.zeros(12, dtype=np.float32)
    custom_attention[4] = 99.0
    custom_attention[6] = 88.0
    engine.update_scores_with_new_tokens(0, attention_weights=custom_attention)

    # 3. Compute eviction
    result = engine.compute_eviction()

    # Budget is 10, total was 12 -> must evict 2
    assert len(result.retained_indices) == 10
    assert len(result.evicted_indices) == 2

    # Sinks (0, 1) MUST be retained
    assert 0 in result.retained_indices
    assert 1 in result.retained_indices

    # Heavy hitters (4, 6) MUST be retained
    assert 4 in result.retained_indices
    assert 6 in result.retained_indices

    # Recent window (9, 10, 11) MUST be retained
    assert 9 in result.retained_indices
    assert 10 in result.retained_indices
    assert 11 in result.retained_indices


def test_streaming_kv_cache_constant_memory_invariant() -> None:
    # Set tight cache budget of 64 tokens
    config = StreamingCacheConfig(
        max_cache_tokens=64,
        num_sink_tokens=4,
        recent_window_tokens=32,
        heavy_hitter_tokens=28,
    )
    manager = StreamingKVCacheManager(config)

    num_heads = 8
    head_dim = 64

    # Simulate generating 500 tokens step-by-step
    np.random.seed(42)
    for _step in range(500):
        new_k = np.random.randn(num_heads, 1, head_dim).astype(np.float32)
        new_v = np.random.randn(num_heads, 1, head_dim).astype(np.float32)

        # Random simulated attention over current cache
        curr_len = manager.current_cache_length
        step_attention = np.random.exponential(scale=1.0, size=curr_len + 1).astype(np.float32)

        cached_k, cached_v, _eviction_result = manager.append_and_evict(
            new_k, new_v, attention_weights=step_attention
        )

        # Invariant: Cache length must NEVER exceed max_cache_tokens
        assert cached_k.shape[-2] <= 64
        assert cached_v.shape[-2] <= 64
        assert manager.current_cache_length <= 64

    # Verify metrics
    metrics = manager.get_metrics()
    assert metrics.total_tokens_seen == 500
    assert metrics.current_cache_size <= 64
    assert metrics.evicted_tokens_count >= 500 - 64
    assert metrics.memory_saving_percentage > 85.0
