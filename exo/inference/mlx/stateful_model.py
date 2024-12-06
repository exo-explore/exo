from collections import OrderedDict

import mlx.nn as nn
from mlx_lm.models.cache import make_prompt_cache

from ..shard import Shard

class StatefulModel(nn.Module):
  def __init__(self, model, max_kv_size: int = 1024, max_caches: int = 2):
    super().__init__()
    self.model = model
    self.max_kv_size = max_kv_size
    self.max_caches = max_caches
    self.caches = OrderedDict()
  
  def init_cache(self, request_id: str):
    cache = make_prompt_cache(self.model)

    if len(self.caches) >= self.max_caches:
      self.caches.popitem(last=False)

    self.caches[request_id] = cache

  def __call__(self, x, request_id: str):
    if request_id not in self.caches:
      self.init_cache(request_id)
    else:
      self.caches.move_to_end(request_id)

    cache = self.caches[request_id]

    y = self.model(x, cache=cache)
    return y
