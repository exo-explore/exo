from exo.inference.shard import Shard
import mlx.core as mx
import mlx.nn as nn
from typing import Optional
import numpy as np


class DummyModel(nn.Module):
  def __init__(self, shard: Optional[Shard] = None):
    self.shard = shard
    self.layers = [
      nn.Linear(8, 128),
      nn.Linear(128, 128),
      nn.Linear(128, 128),
      nn.Linear(128, 128),
      nn.Linear(128, 8),
    ]

    self.n_kv_heads = 4
    self.head_dim = 4

  def __call__(self, x, cache=None):
    if self.shard:
      for layer in self.layers[self.shard.start_layer:self.shard.end_layer + 1]:
        x = layer(x)
      if self.shard.is_last_layer():
        x = x.reshape((1, 2, 4))
    else:
      for layer in self.layers:
        x = layer(x)
      x = x.reshape((1, 2, 4))

    return x


model = DummyModel()
model.save_weights("./test_weights.npz")
n_layers = 5
shard1 = Shard("test", 0, n_layers // 2, n_layers)
sharded_model1 = DummyModel(shard1)
shard2 = Shard("test", n_layers//2 + 1, n_layers - 1, n_layers)
sharded_model2 = DummyModel(shard2)

model.load_weights("./test_weights.npz")
sharded_model1.load_weights("./test_weights.npz")
sharded_model2.load_weights("./test_weights.npz")

fullresp = model(mx.array([1, 2, 3, 4, 5, 6, 7, 8]))
resp1 = sharded_model1(mx.array([1, 2, 3, 4, 5, 6, 7, 8]))
resp2 = sharded_model2(resp1)

assert np.all(np.array(fullresp) == np.array(resp2))
