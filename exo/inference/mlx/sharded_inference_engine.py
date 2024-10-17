import numpy as np
import mlx.core as mx
from ..inference_engine import InferenceEngine
from .sharded_model import StatefulShardedModel
from .sharded_utils import load_shard, get_image_from_str
from ..shard import Shard
from typing import Optional
from exo.download.shard_download import ShardDownloader
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial

class MLXDynamicShardInferenceEngine(InferenceEngine):
  def __init__(self, shard_downloader: ShardDownloader):
    self.shard = None
    self.shard_downloader = shard_downloader
    self.executor = ThreadPoolExecutor(max_workers=1)

  async def infer_prompt(self, request_id: str, shard: Shard, prompt: str, image_str: Optional[str] = None, inference_state: Optional[str] = None) -> (np.ndarray, str, bool):
    await self.ensure_shard(shard)
    step = partial(self.stateful_sharded_model.step, request_id, **await self.get_inputs(prompt, image_str))
    output_data: np.ndarray = np.array(await asyncio.get_running_loop().run_in_executor(self.executor, step))
    return output_data, "", output_data.size == 1 and output_data.item() == self.tokenizer.eos_token_id

  async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[str] = None) -> (np.ndarray, str, bool):
    await self.ensure_shard(shard)
    output_data: np.ndarray = np.array(await asyncio.get_running_loop().run_in_executor(self.executor, self.stateful_sharded_model.step, request_id, mx.array(input_data)))
    return output_data, "", output_data.size == 1 and output_data.item() == self.tokenizer.eos_token_id

  async def ensure_shard(self, shard: Shard):
    if self.shard == shard:
      return

    model_path = await self.shard_downloader.ensure_shard(shard)

    if self.shard != shard:
      loop = asyncio.get_running_loop()
      def load_shard_wrapper(): return asyncio.run(load_shard(model_path, shard))
      model_shard, self.tokenizer = await loop.run_in_executor(self.executor, load_shard_wrapper)
      self.stateful_sharded_model = await loop.run_in_executor(self.executor, StatefulShardedModel, shard, model_shard)
      self.shard = shard
      
  async def get_inputs(self, prompt: str, image_str: Optional[str] = None):
    if image_str:
      image = await get_image_from_str(image_str)
      tokenize = partial(self.tokenizer, text=prompt, images=image, return_tensors="np")
    else:
      tokenize = partial(self.tokenizer, text=prompt, return_tensors="np")
      
    inputs = await asyncio.get_running_loop().run_in_executor(self.executor, tokenize)
    resp = {}
    for attr in ["input_ids", "aspect_ratio_ids", "aspect_ratio_mask"]:
      if attr in inputs: resp[attr] = mx.array(inputs[attr])
      
    pixel_values = inputs.get("pixel_values", None)
    if pixel_values is not None:        
        if type(pixel_values) == list:
          resp["pixel_values"] = [mx.array(pv) for pv in pixel_values[0]]
        else:
          resp["pixel_values"] = mx.array(pixel_values)
    return resp
