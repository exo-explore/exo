# type: ignore
"""Tests for SnapKV KV-cache compression.

These tests exercise the pure-Python / MLX compression logic without
loading a real model.  They run wherever MLX is available.
"""

from copy import deepcopy
from typing import cast

import numpy as np
import pytest

_mlx = pytest.importorskip("mlx.core", reason="mlx not available in this environment")

import mlx.core as mx  # noqa: E402
from mlx_lm.models.cache import KVCache, QuantizedKVCache, RotatingKVCache  # noqa: E402

from exo.worker.engines.mlx.cache import (  # noqa: E402
    SNAPKV_ANCHOR_TOKENS,
    SNAPKV_LOCAL_WINDOW,
    SNAPKV_MAX_CAPACITY_PROMPT,
    SNAPKV_POOLING_KERNEL_SIZE,
    SNAPKV_THRESHOLD,
    cache_length,
    snapkv_compress,
    snapkv_maybe_compress,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_kv_cache(n_layers: int, seq_len: int, heads: int = 2, dim: int = 8) -> list[KVCache]:
    """Create a synthetic KVCache list with predictable key/value tensors."""
    caches: list[KVCache] = []
    for layer_idx in range(n_layers):
        layer = KVCache()
        # Shape: (batch=1, heads, seq_len, dim)
        rng = np.random.default_rng(seed=layer_idx)
        keys = mx.array(rng.standard_normal((1, heads, seq_len, dim)).astype(np.float32))
        values = mx.array(rng.standard_normal((1, heads, seq_len, dim)).astype(np.float32))
        layer.keys = keys
        layer.values = values
        layer.offset = seq_len
        caches.append(layer)
    return caches


def _make_mixed_cache(seq_len: int) -> list[object]:
    """Cache with one KVCache, one QuantizedKVCache, one RotatingKVCache."""
    kv = KVCache()
    rng = np.random.default_rng(seed=0)
    keys = mx.array(rng.standard_normal((1, 2, seq_len, 8)).astype(np.float32))
    values = mx.array(rng.standard_normal((1, 2, seq_len, 8)).astype(np.float32))
    kv.keys = keys
    kv.values = values
    kv.offset = seq_len

    rotating = RotatingKVCache(max_size=seq_len)
    quantized = QuantizedKVCache()
    return [kv, rotating, quantized]


# ---------------------------------------------------------------------------
# snapkv_compress — unit tests
# ---------------------------------------------------------------------------


class TestSnapkvCompressNoOp:
    """Compression must be a no-op when the cache is at or below the threshold."""

    def test_short_cache_not_compressed(self):
        seq_len = 512
        caches = _make_kv_cache(n_layers=2, seq_len=seq_len)
        original_len = cache_length(cast(list, caches))
        new_len = snapkv_compress(cast(list, caches), threshold=2048)
        assert new_len == original_len
        assert cache_length(cast(list, caches)) == original_len

    def test_exactly_at_threshold_not_compressed(self):
        seq_len = 2048
        caches = _make_kv_cache(n_layers=2, seq_len=seq_len)
        new_len = snapkv_compress(cast(list, caches), threshold=2048)
        assert new_len == seq_len

    def test_cache_within_budget_not_compressed(self):
        # anchor=64, local_window=256, max_capacity=2048 → budget=2368
        # seq_len=2100 < threshold=2048 doesn't fire
        seq_len = 2100
        caches = _make_kv_cache(n_layers=1, seq_len=seq_len)
        # Use a threshold of 2000 so compression is attempted, but budget covers it
        new_len = snapkv_compress(
            cast(list, caches),
            threshold=2000,
            anchor_tokens=64,
            local_window=256,
            max_capacity_prompt=2048,
        )
        # budget = 64+256+2048 = 2368 > seq_len=2100, so still no-op
        assert new_len == seq_len


class TestSnapkvCompressReduces:
    """Compression must reduce cache length when context is long."""

    def test_long_cache_is_compressed(self):
        seq_len = 4096
        caches = _make_kv_cache(n_layers=2, seq_len=seq_len)
        new_len = snapkv_compress(
            cast(list, caches),
            threshold=2048,
            anchor_tokens=64,
            local_window=256,
            max_capacity_prompt=512,
        )
        expected_max = 64 + 256 + 512  # = 832
        assert new_len <= expected_max
        assert new_len < seq_len

    def test_compression_ratio_at_16k(self):
        seq_len = 16384
        caches = _make_kv_cache(n_layers=1, seq_len=seq_len)
        new_len = snapkv_compress(
            cast(list, caches),
            threshold=2048,
            anchor_tokens=64,
            local_window=256,
            max_capacity_prompt=512,
        )
        budget = 64 + 256 + 512
        assert new_len == budget
        # Compression ratio should be substantial
        ratio = new_len / seq_len
        assert ratio < 0.1  # < 10% of original

    def test_compressed_length_equals_cache_length(self):
        """Return value must agree with cache_length() after compression."""
        seq_len = 8192
        caches = _make_kv_cache(n_layers=3, seq_len=seq_len)
        returned = snapkv_compress(
            cast(list, caches),
            threshold=2048,
            anchor_tokens=64,
            local_window=256,
            max_capacity_prompt=512,
        )
        assert returned == cache_length(cast(list, caches))


class TestSnapkvCompressStructure:
    """Verify that anchor + important + recent token structure is respected."""

    def test_anchor_tokens_kept(self):
        """First anchor_tokens positions must always be present in compressed output."""
        seq_len = 4096
        anchor = 32
        local_window = 128
        max_capacity = 256
        n_layers = 1
        caches = _make_kv_cache(n_layers=n_layers, seq_len=seq_len, heads=1, dim=4)
        layer_before = deepcopy(caches[0])

        snapkv_compress(
            cast(list, caches),
            threshold=512,
            anchor_tokens=anchor,
            local_window=local_window,
            max_capacity_prompt=max_capacity,
        )

        # First anchor rows in compressed cache must match original first anchor rows.
        compressed_keys = caches[0].keys
        original_keys = layer_before.keys
        assert compressed_keys is not None and original_keys is not None
        # Shape check: (1, heads, new_len, dim)
        assert compressed_keys.shape[2] <= anchor + local_window + max_capacity
        # Anchor rows must be identical
        np.testing.assert_array_almost_equal(
            np.array(compressed_keys[..., :anchor, :]),
            np.array(original_keys[..., :anchor, :]),
            decimal=5,
        )

    def test_recent_tokens_kept(self):
        """Last local_window positions must always be present in compressed output."""
        seq_len = 4096
        anchor = 32
        local_window = 64
        max_capacity = 128
        caches = _make_kv_cache(n_layers=1, seq_len=seq_len, heads=1, dim=4)
        layer_before = deepcopy(caches[0])

        snapkv_compress(
            cast(list, caches),
            threshold=512,
            anchor_tokens=anchor,
            local_window=local_window,
            max_capacity_prompt=max_capacity,
        )

        compressed_keys = caches[0].keys
        original_keys = layer_before.keys
        assert compressed_keys is not None and original_keys is not None
        new_len = compressed_keys.shape[2]
        # Last local_window rows in compressed cache must match original last rows
        np.testing.assert_array_almost_equal(
            np.array(compressed_keys[..., new_len - local_window :, :]),
            np.array(original_keys[..., seq_len - local_window :, :]),
            decimal=5,
        )


class TestSnapkvCompressMixedLayerTypes:
    """Non-KVCache layers must be left untouched."""

    def test_rotating_and_quantized_untouched(self):
        seq_len = 4096
        mixed = _make_mixed_cache(seq_len)
        kv_before_offset = cast(KVCache, mixed[0]).offset
        rotating_before = mixed[1]
        quantized_before = mixed[2]

        snapkv_compress(
            cast(list, mixed),
            threshold=512,
            anchor_tokens=16,
            local_window=32,
            max_capacity_prompt=64,
        )

        # KVCache layer must have been compressed
        assert cast(KVCache, mixed[0]).offset < kv_before_offset
        # RotatingKVCache and QuantizedKVCache must be unchanged objects
        assert mixed[1] is rotating_before
        assert mixed[2] is quantized_before


class TestSnapkvPoolingKernel:
    """Pooling kernel size must not change the output shape (only the token selection)."""

    def test_kernel_1_matches_no_smoothing(self):
        """With kernel=1, result must match compress_kv_cache output shape."""
        seq_len = 4096
        caches_k1 = _make_kv_cache(n_layers=1, seq_len=seq_len)
        caches_k5 = deepcopy(caches_k1)

        len_k1 = snapkv_compress(
            cast(list, caches_k1),
            threshold=512,
            anchor_tokens=16,
            local_window=32,
            max_capacity_prompt=64,
            pooling_kernel_size=1,
        )
        len_k5 = snapkv_compress(
            cast(list, caches_k5),
            threshold=512,
            anchor_tokens=16,
            local_window=32,
            max_capacity_prompt=64,
            pooling_kernel_size=5,
        )
        # Both should produce the same budget-capped length
        assert len_k1 == len_k5 == 16 + 64 + 32

    def test_large_kernel_still_compresses(self):
        seq_len = 4096
        caches = _make_kv_cache(n_layers=1, seq_len=seq_len)
        new_len = snapkv_compress(
            cast(list, caches),
            threshold=512,
            anchor_tokens=16,
            local_window=32,
            max_capacity_prompt=64,
            pooling_kernel_size=15,
        )
        assert new_len <= 16 + 64 + 32


# ---------------------------------------------------------------------------
# snapkv_maybe_compress — env-var gating
# ---------------------------------------------------------------------------


class TestSnapkvMaybeCompress:
    def test_disabled_by_default(self):
        """snapkv_maybe_compress must be a no-op when EXO_SNAPKV is not set."""
        seq_len = 8192
        caches = _make_kv_cache(n_layers=2, seq_len=seq_len)
        original_len = cache_length(cast(list, caches))
        # Patch module-level flag to False regardless of env
        import exo.worker.engines.mlx.cache as cache_mod

        original_flag = cache_mod._SNAPKV_ENABLED
        try:
            cache_mod._SNAPKV_ENABLED = False
            result = snapkv_maybe_compress(cast(list, caches))
        finally:
            cache_mod._SNAPKV_ENABLED = original_flag

        assert result == original_len
        assert cache_length(cast(list, caches)) == original_len

    def test_enabled_compresses(self):
        """snapkv_maybe_compress must compress when EXO_SNAPKV=1 and cache > threshold."""
        seq_len = SNAPKV_THRESHOLD + 1000
        caches = _make_kv_cache(n_layers=2, seq_len=seq_len)
        original_len = cache_length(cast(list, caches))

        import exo.worker.engines.mlx.cache as cache_mod

        original_flag = cache_mod._SNAPKV_ENABLED
        try:
            cache_mod._SNAPKV_ENABLED = True
            result = snapkv_maybe_compress(cast(list, caches))
        finally:
            cache_mod._SNAPKV_ENABLED = original_flag

        assert result < original_len
        assert result == cache_length(cast(list, caches))

    def test_enabled_no_op_short_cache(self):
        """Even when enabled, must not compress a short cache."""
        seq_len = 64
        caches = _make_kv_cache(n_layers=1, seq_len=seq_len)

        import exo.worker.engines.mlx.cache as cache_mod

        original_flag = cache_mod._SNAPKV_ENABLED
        try:
            cache_mod._SNAPKV_ENABLED = True
            result = snapkv_maybe_compress(cast(list, caches))
        finally:
            cache_mod._SNAPKV_ENABLED = original_flag

        assert result == seq_len


# ---------------------------------------------------------------------------
# Default-constant sanity checks
# ---------------------------------------------------------------------------


class TestSnapkvConstants:
    def test_threshold_default(self):
        assert SNAPKV_THRESHOLD == 2048

    def test_anchor_default(self):
        assert SNAPKV_ANCHOR_TOKENS == 64

    def test_local_window_default(self):
        assert SNAPKV_LOCAL_WINDOW == 256

    def test_max_capacity_default(self):
        assert SNAPKV_MAX_CAPACITY_PROMPT == 2048

    def test_pooling_kernel_default(self):
        assert SNAPKV_POOLING_KERNEL_SIZE == 5

    def test_budget_sane(self):
        budget = SNAPKV_ANCHOR_TOKENS + SNAPKV_LOCAL_WINDOW + SNAPKV_MAX_CAPACITY_PROMPT
        assert budget < SNAPKV_THRESHOLD * 3  # sanity: budget shouldn't be absurdly large
