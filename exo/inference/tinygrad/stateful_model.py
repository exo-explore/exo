from tinygrad import Tensor, Variable 
from collections import OrderedDict
from typing import List, Optional

def create_kv_cache(x: Tensor, max_context: int, n_kv_heads: int, head_dim: int):
  cache_kv = Tensor.zeros(2, x.shape[0], max_context, n_kv_heads, head_dim, dtype=x.dtype).contiguous().realize()
  if isinstance(x.device, tuple):
    # TODO: instead of specifying how to shard, it can follow how xk and xv are being sharded
    cache_kv.shard_((x.device), axis=3 if getenv("SHARD_KVCACHE") else None).realize()
  return cache_kv.realize()

class ModelState:
  cache: List[Tensor]
  start: int 
  def __init__(self, cache: List[Tensor], start: int = 0):
    self.cache = cache
    self.start = start

def make_prompt_state(x, model, shard):
  cache = [create_kv_cache(x, model.layers[i].attention.max_context, model.layers[i].attention.n_kv_heads, model.layers[i].attention.head_dim) for i in range(shard.start_layer, shard.end_layer + 1)]

  return ModelState(cache)
