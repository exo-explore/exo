from typing import Dict, Tuple, Optional
from collections import OrderedDict

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import make_prompt_cache
import numpy as np

from ..shard import Shard
class StatefulModel(nn.Module):
  def __init__(self, model, max_kv_size: int = 1024, max_caches: int = 2):
    super().__init__()
    self.model = model
    self.max_kv_size = max_kv_size
    self.max_caches = max_caches
    self.caches = OrderedDict()
  
  def __call__(self, x, request_id: Optional[str] = None, use_cache: bool = True):
    #print(f"StatefulModel in <- {x}")
    if use_cache and request_id is not None:
      if request_id not in self.caches:
        self.init_cache(request_id)
      else:
        self.caches.move_to_end(request_id)

      cache = mx.array(self.caches[request_id])
      y = self.model(x, cache=cache)
    else:
      y = self.model(x)
    #print(f"StatefulModel out -> {y}")
    return y
    
