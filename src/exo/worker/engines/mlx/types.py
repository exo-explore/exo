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

KVCacheType = Sequence[
    KVCache
    | RotatingKVCache
    | QuantizedKVCache
    | ArraysCache
    | CacheList
    | DeepseekV4Cache
]


class Model(nn.Module):
    layers: list[nn.Module]

    def __call__(
        self,
        x: mx.array,
        cache: KVCacheType | None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array: ...
