from typing import Optional, Tuple, TYPE_CHECKING
import numpy as np
import random
import string
import asyncio
import json
from exo.inference.inference_engine import InferenceEngine
from exo.inference.shard import Shard
def random_string(length: int):
  return ''.join([random.choice(string.ascii_lowercase) for i in range(length)])
  

class DummyInferenceEngine(InferenceEngine):
  def __init__(self):
    self.shard = None
    self.vocab_size = 1000
    self.hidden_size = 256
    self.eos_token_id = 0
    self.latency_mean = 0.1
    self.latency_stddev = 0.02

  async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
    return np.random.randint(1, self.vocab_size, size=(1, len(prompt.split())))
  
  async def sample(self, x: np.ndarray) -> np.ndarray:
    return np.random.randint(1, self.vocab_size)

  async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
    return ' '.join([random_string(np.random.randint(1, 34)) for token in tokens])

  async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[str] = None) -> np.ndarray:
    await self.ensure_shard(shard)
    sequence_length = input_data.shape[0 if self.shard.is_first_layer() else 1]
    output = np.random.random(size=(1, sequence_length, self.vocab_size if self.shard.is_last_layer() else self.hidden_size))
    return output

  async def ensure_shard(self, shard: Shard):
    if self.shard == shard:
      return
    # Simulate shard loading without making any API calls
    await asyncio.sleep(0.1)  # Simulate a short delay
    self.shard = shard
    print(f"DummyInferenceEngine: Simulated loading of shard {shard.model_id}")
