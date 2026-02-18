import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache

# These are wrapper functions to fix the fact that mlx is not strongly typed in the same way that EXO is.
# For example - MLX has no guarantee of the interface that nn.Module will expose. But we need a guarantee that it has a __call__() function


class Model(nn.Module):
    layers: list[nn.Module]

    def __call__(
        self,
        x: mx.array,
        cache: list[KVCache] | None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array: ...
