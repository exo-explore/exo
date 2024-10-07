import time
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

class MLXDynamicShardInferenceEngine(InferenceEngine):
  def __init__(self, shard_downloader: ShardDownloader):
    self.shard = None
    self.shard_downloader = shard_downloader
    self.executor = ThreadPoolExecutor(max_workers=1)

  async def infer_prompt(self, request_id: str, shard: Shard, prompt: str, image_str: Optional[str] = None, inference_state: Optional[str] = None) -> (np.ndarray, str, bool):
    await self.ensure_shard(shard)
    loop = asyncio.get_running_loop()
    if image_str:
      image = await get_image_from_str(image_str)
      inputs = await loop.run_in_executor(self.executor, self.tokenizer, prompt, image, return_tensors="np")
      pixel_values = mx.array(inputs["pixel_values"])
      input_ids = mx.array(inputs["input_ids"])
      output_data: np.ndarray = np.array(await loop.run_in_executor(self.executor, self.stateful_sharded_model.step, request_id, input_ids, pixel_values))
    else:
      input_ids = mx.array(await loop.run_in_executor(self.executor, self.tokenizer.encode, prompt))
      output_data: np.ndarray = np.array(await loop.run_in_executor(self.executor, self.stateful_sharded_model.step, request_id, input_ids))
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

  def _benchmark_tflops(n, dtype='f32', num_iterations=10):
    
    if dtype == 'f32':
        A = mx.random.normal((n, n), dtype=mx.float32)
        B = mx.random.normal((n, n), dtype=mx.float32)
    elif dtype == 'f16':
        A = mx.random.normal((n, n), dtype=mx.float16)
        B = mx.random.normal((n, n), dtype=mx.float16)
    elif dtype == 'int8':
        A = (mx.random.randint(-128, 127, (n, n), dtype=mx.int8)).astype(mx.float32)
        B = (mx.random.randint(-128, 127, (n, n), dtype=mx.int8)).astype(mx.float32)
    else:
        raise ValueError("Unsupported data type. Use 'f32', 'f16', or 'int8'.")

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        C = mx.matmul(A, B)
        mx.synchronize()
    elapsed_time = time.perf_counter() - start_time

    flops_per_iteration = 2 * (n ** 3)
    total_flops = flops_per_iteration * num_iterations
    tflops = (total_flops / elapsed_time) / 1e12

    return float(f"{tflops:.2f}")
  
def benchmark_tflops(self):
    n = 2**8
    fp32=self._benchmark_tflops(n)
    fp16=self._benchmark_tflops(n, dtype='f16')
    int8=self._benchmark_tflops(n, dtype='int8')

    return (fp32, fp16, int8)