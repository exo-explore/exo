"""Shared types for MLX-related functionality."""

from collections.abc import Sequence

from mlx_lm.models.cache import (
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
)

# Type alias for KV cache - matches make_kv_cache return type
# This list contains one cache entry per transformer layer
# Using Sequence (covariant) instead of list (invariant) to allow
# list[KVCache] to be assignable to KVCacheType
KVCacheType = Sequence[KVCache | RotatingKVCache | QuantizedKVCache]
