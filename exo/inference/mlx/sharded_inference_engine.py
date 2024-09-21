import numpy as np
import mlx.core as mx
from ..inference_engine import InferenceEngine
from .sharded_model import StatefulShardedModel
from .sharded_utils import load_shard, get_image_from_str
from ..shard import Shard
from typing import Optional, List
from exo.download.shard_download import ShardDownloader
import asyncio
from functools import partial
from concurrent.futures import ThreadPoolExecutor


class MLXDynamicShardInferenceEngine(InferenceEngine):
  def __init__(self, shard_downloader: ShardDownloader):
    self.shard = None
    self.shard_downloader = shard_downloader
    self.executor = ThreadPoolExecutor(max_workers=1)
    self.request_eos = dict() # TODO: this keeps growing, make it LRU

  async def infer_prompt(self, request_ids: List[str], shard: Shard, prompts: List[str], image_strs: Optional[List[str]] = None, inference_state: Optional[str] = None) -> (np.ndarray, str, bool):
    await self.ensure_shard(shard)
    loop = asyncio.get_running_loop()
    if image_strs:
      images = await get_image_from_str(image_strs)
      tokenize = partial(self.tokenizer, prompts, images, return_tensors="np", padding=True)
      inputs = await loop.run_in_executor(self.executor, tokenize)
      pixel_values = mx.array(inputs["pixel_values"])
      input_ids = mx.array(inputs["input_ids"])
      output_data: np.ndarray = np.array(await loop.run_in_executor(self.executor, self.stateful_sharded_model.step, request_ids, input_ids, pixel_values))
    else:
      tokenize = partial(self.tokenizer._tokenizer(prompts, padding=True, truncation=True)["input_ids"])
      input_ids = mx.array(await loop.run_in_executor(self.executor, tokenize))
      output_data: np.ndarray = np.array(await loop.run_in_executor(self.executor, self.stateful_sharded_model.step, request_ids, input_ids))
    return output_data, "", self.get_is_finished(request_ids, output_data)

  async def infer_tensor(self, request_ids: List[str], shard: Shard, input_data: np.ndarray, inference_state: Optional[str] = None) -> (np.ndarray, str, bool):
    await self.ensure_shard(shard)
    output_data: np.ndarray = np.array(await asyncio.get_running_loop().run_in_executor(self.executor, self.stateful_sharded_model.step, request_ids, mx.array(input_data)))
    return output_data, "", self.get_is_finished(request_ids, output_data)

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

  def get_is_finished(self, request_ids: List[str], output_data: np.ndarray):
    if output_data.ndim != 2:
      return False
    comb_id = "_".join(request_ids)
    _request_eos = self.request_eos.setdefault(comb_id, np.zeros(len(request_ids), dtype=bool))
    _request_eos |= (output_data == self.tokenizer.eos_token_id).flatten()
    return all(_request_eos)