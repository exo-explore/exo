import numpy as np
import mlx.core as mx
from ..inference_engine import InferenceEngine
from .sharded_model import StatefulShardedModel
from .sharded_utils import load_shard, get_image_from_str
from ..shard import Shard
from typing import Optional


class MLXDynamicShardInferenceEngine(InferenceEngine):
  def __init__(self):
    self.shard = None

  async def infer_prompt(self, request_id: str, shard: Shard, prompt: str, image_str: Optional[str] = None, inference_state: Optional[str] = None) -> (np.ndarray, str, bool):
    await self.ensure_shard(shard)
    if image_str:
      image = get_image_from_str(image_str)
      inputs = self.tokenizer(prompt, image, return_tensors="np")
      pixel_values = mx.array(inputs["pixel_values"])
      input_ids = mx.array(inputs["input_ids"])
      output_data: np.ndarray = np.array(self.stateful_sharded_model.step(request_id, input_ids, pixel_values))
    else:
      output_data: np.ndarray = np.array(self.stateful_sharded_model.step(request_id, mx.array(self.tokenizer.encode(prompt))))
    return output_data, "", output_data.size == 1 and output_data.item() == self.tokenizer.eos_token_id

  async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[str] = None) -> (np.ndarray, str, bool):
    await self.ensure_shard(shard)
    output_data: np.ndarray = np.array(self.stateful_sharded_model.step(request_id, mx.array(input_data)))
    return output_data, "", output_data.size == 1 and output_data.item() == self.tokenizer.eos_token_id

  async def ensure_shard(self, shard: Shard):
    if self.shard == shard:
      return

    model_shard, self.tokenizer = await load_shard(shard.model_id, shard)
    self.stateful_sharded_model = StatefulShardedModel(shard, model_shard)
    self.shard = shard
