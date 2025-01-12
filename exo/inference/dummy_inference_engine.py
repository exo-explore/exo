from typing import Optional, Tuple, TYPE_CHECKING
import numpy as np
from exo.inference.inference_engine import InferenceEngine
from exo.inference.shard import Shard
from exo.inference.tokenizers import DummyTokenizer

class DummyInferenceEngine(InferenceEngine):
  def __init__(self):
    self.shard = None
    self.vocab_size = 1000
    self.hidden_size = 256
    self.eos_token_id = 0
    self.latency_mean = 0.1
    self.latency_stddev = 0.02
    self.num_generate_dummy_tokens = 10
    self.tokenizer = DummyTokenizer()

  async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
    return np.array(self.tokenizer.encode(prompt))
  
  async def sample(self, x: np.ndarray, temp: float = 0.0, top_p: float = 1.0) -> np.ndarray:
    if x[0] > self.num_generate_dummy_tokens: return np.array([self.tokenizer.eos_token_id])
    return x

  async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
    return self.tokenizer.decode(tokens)

  async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict]]:
    await self.ensure_shard(shard)
    return input_data + 1 if self.shard.is_last_layer() else input_data, None

  async def ensure_shard(self, shard: Shard):
    if self.shard == shard: return
    self.shard = shard
  
  async def load_checkpoint(self, shard: Shard, path: str):
    await self.ensure_shard(shard)
