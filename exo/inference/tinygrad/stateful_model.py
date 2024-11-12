from tinygrad import Tensor, Variable 
from collections import OrderedDict

def create_kv_cache(x: Tensor, max_context: int, n_kv_heads: int, head_dim: int):
  cache_kv = Tensor.zeros(2, x.shape[0], max_context, n_kv_heads, head_dim, dtype=x.dtype).contiguous().realize()
  if isinstance(x.device, tuple):
    # TODO: instead of specifying how to shard, it can follow how xk and xv are being sharded
    cache_kv.shard_((x.device), axis=3 if getenv("SHARD_KVCACHE") else None).realize()
  return cache_kv.realize()

class StatefulModel:
  def __init__(self, model, max_caches: int = 2):
    super().__init__()
    self.model = model
    self.max_caches = max_caches
    self.caches = OrderedDict()
 
  def init_cache(self, x: Tensor, request_id: str):
    cache = [create_kv_cache(x, self.model.layers[i].attention.max_context, self.model.layers[i].attention.n_kv_heads, self.model.layers[i].attention.head_dim) for i in range(self.model.shard.start_layer, self.model.shard.end_layer + 1)]
    if len(self.caches) >= self.max_caches:
      self.caches.popitem(last=False)

    self.caches[request_id] = cache

  def __call__(self, x: Tensor, start_pos: Variable, request_id: str): 
    h = self.model.embed(x)
    if request_id not in self.caches:
      self.init_cache(h, request_id)
    else:
      self.caches.move_to_end(request_id)
    if h.shape[0:2] == (1, 1) and self.model.forward_jit is not None:
      return self.model.forward_jit(h, Variable("start_pos", 0, self.model.max_context).bind(start_pos), cache=self.caches[request_id])
    return self.model.forward(h, start_pos, cache=self.caches[request_id])

