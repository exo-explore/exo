from typing import Optional
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache


class IdentityBlock(nn.Module):
  def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[KVCache] = None) -> mx.array:
    return x
