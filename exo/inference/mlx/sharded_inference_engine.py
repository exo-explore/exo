import numpy as np
import mlx.core as mx
from ..inference_engine import InferenceEngine
from .sharded_model import StatefulShardedModel
from .sharded_utils import load_shard, get_image_from_str
from ..shard import Shard
from typing import Optional
from exo.download.shard_download import ShardDownloader
import threading
from concurrent.futures import ThreadPoolExecutor
import asyncio

class MLXDynamicShardInferenceEngine(InferenceEngine):
  def __init__(self, shard_downloader: ShardDownloader):
    self.shard = None
    self.shard_downloader = shard_downloader
    self.model_lock = threading.Lock()
    self.executor = ThreadPoolExecutor(max_workers=1)

  async def infer_prompt(self, request_id: str, shard: Shard, prompt: str, image_str: Optional[str] = None, inference_state: Optional[str] = None) -> (np.ndarray, str, bool):
    await self.ensure_shard(shard)
    if image_str:
      image = await get_image_from_str(image_str)
      inputs = self.tokenizer(prompt, image, return_tensors="np")
      pixel_values = mx.array(inputs["pixel_values"])
      input_ids = mx.array(inputs["input_ids"])
      output_data = await self._run_inference(request_id, input_ids, pixel_values)
    else:
      input_ids = mx.array(self.tokenizer.encode(prompt))
      output_data = await self._run_inference(request_id, input_ids)
    return output_data, "", output_data.size == 1 and output_data.item() == self.tokenizer.eos_token_id

  async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[str] = None) -> (np.ndarray, str, bool):
    await self.ensure_shard(shard)
    input_tensor = mx.array(input_data)
    output_data = await self._run_inference(request_id, input_tensor)
    return output_data, "", output_data.size == 1 and output_data.item() == self.tokenizer.eos_token_id

  async def _run_inference(self, request_id: str, *args):
    with self.model_lock:
      return await asyncio.get_event_loop().run_in_executor(
        self.executor,
        lambda: np.array(self.stateful_sharded_model.step(request_id, *args))
      )

  async def ensure_shard(self, shard: Shard):
    if self.shard == shard:
      return

    model_path = await self.shard_downloader.ensure_shard(shard)

    with self.model_lock:
      if self.shard != shard:
        model_shard, self.tokenizer = await load_shard(model_path, shard)
        self.stateful_sharded_model = StatefulShardedModel(shard, model_shard)
        self.shard = shard
