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

class StatefulModel:
  def __init__(self, model, max_states: int = 2):
    super().__init__()
    self.model = model
    self.max_states = max_states
    self.states = OrderedDict()
 
  def init_cache(self, x: Tensor, request_id: str):
    cache = [create_kv_cache(x, self.model.layers[i].attention.max_context, self.model.layers[i].attention.n_kv_heads, self.model.layers[i].attention.head_dim) for i in range(self.model.shard.start_layer, self.model.shard.end_layer + 1)]
    if len(self.states) >= self.max_states:
      self.states.popitem(last=False)

    self.states[request_id] = ModelState(cache)

  def __call__(self, x: Tensor, request_id: Optional[str] = None, use_cache: bool = True): 
    h = self.model.embed(x)
    #print(f"StatefulModel in <- {h}")
    if use_cache and request_id is not None:
      if request_id not in self.states:
        self.init_cache(h, request_id)
      else:
        self.states.move_to_end(request_id)
      out = self.model.forward(h, self.states[request_id].start, cache=self.states[request_id].cache)
      self.states[request_id].start += h.shape[1]
    else:
      out = self.model.forward(h, 0)
    #print(f"StatefulModel out -> {out}")
    return out

