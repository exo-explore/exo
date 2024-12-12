from tinygrad import Tensor, Variable 
from collections import OrderedDict
from typing import List, Optional

def create_kv_cache(x: Tensor, layer):
  cache_kv = Tensor.zeros(2, x.shape[0], layer.max_context, layer.n_kv_heads, layer.head_dim, dtype=x.dtype).contiguous().realize()
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

def make_prompt_state(x: Tensor, model):
  cache = [create_kv_cache(x, l.attention) for l in model.layers]

  return ModelState(cache)
