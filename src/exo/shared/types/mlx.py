"""Shared types for MLX-related functionality."""

from collections.abc import Sequence

from mlx_lm.models.cache import (
    KVCache,
    MambaCache,
    QuantizedKVCache,
    RotatingKVCache,
)

# This list contains one cache entry per transformer layer
KVCacheType = Sequence[KVCache | RotatingKVCache | QuantizedKVCache | MambaCache]
