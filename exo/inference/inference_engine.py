import numpy as np
import os

from typing import Tuple, Optional
from abc import ABC, abstractmethod
from .shard import Shard


class InferenceEngine(ABC):
  @abstractmethod
  
async def infer_prompt(self, request_id: str, shard: Shard, prompts: list[str], image_strs: Optional[list[str]] = None, inference_states: Optional[list[str]] = None) -> (list[np.ndarray], list[str], list[bool]):
    batch_results = []
    batch_states = []
    batch_dones = []

    for idx, prompt in enumerate(prompts):
        # Call infer_prompt for each prompt individually and collect results
        image_str = image_strs[idx] if image_strs else None
        inference_state = inference_states[idx] if inference_states else None
        
        result, state, done = await self._infer_prompt_single(request_id, shard, prompt, image_str, inference_state)
        batch_results.append(result)
        batch_states.append(state)
        batch_dones.append(done)

    return batch_results, batch_states, batch_dones


  @abstractmethod
 
async def infer_tensor(self, request_id: str, shard: Shard, input_data: list[np.ndarray], inference_states: Optional[list[str]] = None) -> Tuple[list[np.ndarray], list[str], list[bool]]:
    batch_results = []
    batch_states = []
    batch_dones = []

    for idx, input_tensor in enumerate(input_data):
        inference_state = inference_states[idx] if inference_states else None
        
        result, state, done = await self._infer_tensor_single(request_id, shard, input_tensor, inference_state)
        batch_results.append(result)
        batch_states.append(state)
        batch_dones.append(done)

    return batch_results, batch_states, batch_dones

async def _infer_prompt_single(self, request_id: str, shard: Shard, prompt: str, image_str: Optional[str] = None, inference_state: Optional[str] = None) -> (np.ndarray, str, bool):
    pass

async def _infer_tensor_single(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[str] = None) -> Tuple[np.ndarray, str, bool]:
    pass


def get_inference_engine(inference_engine_name: str, shard_downloader: 'ShardDownloader'):
  if inference_engine_name == "mlx":
    from exo.inference.mlx.sharded_inference_engine import MLXDynamicShardInferenceEngine

    return MLXDynamicShardInferenceEngine(shard_downloader)
  elif inference_engine_name == "tinygrad":
    from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine
    import tinygrad.helpers
    tinygrad.helpers.DEBUG.value = int(os.getenv("TINYGRAD_DEBUG", default="0"))

    return TinygradDynamicShardInferenceEngine(shard_downloader)
  else:
    raise ValueError(f"Inference engine {inference_engine_name} not supported")
