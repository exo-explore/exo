"""Shared types for MLX-related functionality."""

from collections.abc import Sequence

from mlx import core as mx
from mlx import nn as nn
from mlx_lm.models.cache import (
    ArraysCache,
    CacheList,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
)
from mlx_lm.models.deepseek_v4 import DeepseekV4Cache
from mlx_vlm.models.cache import ArraysCache as MLXVLMArrayCache
from mlx_vlm.models.cache import KVCache as MLXVLMKVCache
from mlx_vlm.models.cache import QuantizedKVCache as MLXVLMQuantizedKVCache

# This list contains one cache entry per transformer layer
KVCacheType = Sequence[
    KVCache
    | RotatingKVCache
    | QuantizedKVCache
    | ArraysCache
    | CacheList
    | DeepseekV4Cache
    | MLXVLMArrayCache
    | MLXVLMKVCache
    | MLXVLMQuantizedKVCache
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
