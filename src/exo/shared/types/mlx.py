"""Shared types for MLX-related functionality."""

from mlx_lm.models.cache import (
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
)

# Type alias for KV cache - matches make_kv_cache return type
# This list contains one cache entry per transformer layer
KVCacheType = list[KVCache | RotatingKVCache | QuantizedKVCache]
