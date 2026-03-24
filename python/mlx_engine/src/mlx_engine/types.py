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
# from vllm_engine.kv_cache import TorchKVCache

MLXCacheType = Sequence[
    KVCache | RotatingKVCache | QuantizedKVCache | ArraysCache | CacheList
]

KVCacheType = MLXCacheType  # | TorchKVCache


# Model is a wrapper function to fix the fact that mlx is not strongly typed in the same way that EXO is.
# For example - MLX has no guarantee of the interface that nn.Module will expose. But we need a guarantee that it has a __call__() function
class Model(nn.Module):
    layers: list[nn.Module]

    def __call__(
        self,
        x: mx.array,
        cache: MLXCacheType | None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array: ...
