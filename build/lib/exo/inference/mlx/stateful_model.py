from typing import Dict, Tuple, Optional
from collections import OrderedDict

import mlx.core as mx
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
    kv_heads = ([self.model.n_kv_heads]*len(self.model.layers) if isinstance(self.model.n_kv_heads, int) else self.model.n_kv_heads)
    # if self.max_kv_size is not None:
      # cache = [RotatingKVCache(self.model.head_dim, n, max_size=self.max_kv_size, keep=4) for n in kv_heads]
      # cache = [KVCache(self.model.head_dim, n) for n in kv_heads]
    # else:
      # cache = [KVCache(self.model.head_dim, n) for n in kv_heads]
    cache = make_prompt_cache(self.model)

    if len(self.caches) >= self.max_caches:
      self.caches.popitem(last=False)

    self.caches[request_id] = cache

  def __call__(self, x, request_id: str, inference_state: Optional[dict] = None):
    if self.model.model_type !='StableDiffusionPipeline':
      if request_id not in self.caches:
        self.init_cache(request_id)
      else:
        self.caches.move_to_end(request_id)

      cache = self.caches[request_id]

      y = self.model(x, cache=cache)
    else:
      y, inference_state = self.model(x, **inference_state)
    return y, inference_state
    
