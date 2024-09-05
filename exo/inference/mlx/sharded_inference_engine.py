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
    self.inference_queue = asyncio.Queue()
    self._worker_task = None

  async def _ensure_worker(self):
    if self._worker_task is None:
      self._worker_task = asyncio.create_task(self._inference_worker())

  async def infer_prompt(self, request_id: str, shard: Shard, prompt: str, image_str: Optional[str] = None, inference_state: Optional[str] = None) -> (np.ndarray, str, bool):
    await self.ensure_shard(shard)
    await self._ensure_worker()

    if image_str:
      image = await get_image_from_str(image_str)
      inputs = self.tokenizer(prompt, image, return_tensors="np")
      pixel_values = mx.array(inputs["pixel_values"])
      input_ids = mx.array(inputs["input_ids"])
      output_data = await self._queue_inference(request_id, input_ids, pixel_values)
    else:
      input_ids = mx.array(self.tokenizer.encode(prompt))
      output_data = await self._queue_inference(request_id, input_ids)
    return output_data, "", output_data.size == 1 and output_data.item() == self.tokenizer.eos_token_id

  async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[str] = None) -> (np.ndarray, str, bool):
    await self.ensure_shard(shard)
    await self._ensure_worker()

    input_tensor = mx.array(input_data)
    output_data = await self._queue_inference(request_id, input_tensor)
    return output_data, "", output_data.size == 1 and output_data.item() == self.tokenizer.eos_token_id

  async def _queue_inference(self, request_id: str, *args):
    future = asyncio.get_running_loop().create_future()
    await self.inference_queue.put((future, request_id, args))
    return await future

  async def _inference_worker(self):
    while True:
      future, request_id, args = await self.inference_queue.get()
      try:
        result = await asyncio.get_running_loop().run_in_executor(
          self.executor,
          lambda: np.array(self.stateful_sharded_model.step(request_id, *args))
        )
        future.set_result(result)
      except Exception as e:
        future.set_exception(e)
      finally:
        self.inference_queue.task_done()

  async def ensure_shard(self, shard: Shard):
    if self.shard == shard:
      return

    model_path = await self.shard_downloader.ensure_shard(shard)

    with self.model_lock:
      if self.shard != shard:
        model_shard, self.tokenizer = await load_shard(model_path, shard)
        self.stateful_sharded_model = StatefulShardedModel(shard, model_shard)
        self.shard = shard
