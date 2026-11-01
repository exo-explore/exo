"""Shared types for MLX-related functionality."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from mlx import core as mx
from mlx import nn as nn
from mlx_lm.models.cache import (
    ArraysCache,
    CacheList,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
)

from exo.worker.engines.mlx.models.deepseek_v4.model import DeepseekV4Cache


@runtime_checkable
class SnapshotCache(Protocol):
    """A model's own per-layer cache that can't be trimmed back token by token (e.g. one
    that keeps a sliding window and pooled state). The prefix cache keeps snapshots of it
    taken at known token counts and restores one of those instead of trimming."""

    offset: int

    @property
    def nbytes(self) -> int: ...

    def prefix_cache_snapshot(self) -> "SnapshotCache":
        """An independent copy of the cache as it is now, with its arrays evaluated."""
        ...


# This list contains one cache entry per transformer layer
KVCacheType = Sequence[
    KVCache
    | RotatingKVCache
    | QuantizedKVCache
    | ArraysCache
    | CacheList
    | DeepseekV4Cache
    | SnapshotCache
]


# Model is a wrapper function to fix the fact that mlx is not strongly typed in the same way that EXO is.
# For example - MLX has no guarantee of the interface that nn.Module will expose. But we need a guarantee that it has a __call__() function
class Model(nn.Module):
    layers: list[nn.Module]

    def __call__(
        self,
        x: mx.array,
        cache: KVCacheType | None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array: ...
