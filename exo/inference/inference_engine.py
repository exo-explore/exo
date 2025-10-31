import numpy as np
import os
from exo.helpers import DEBUG  # Make sure to import DEBUG

from typing import Tuple, Optional
from abc import ABC, abstractmethod
from .shard import Shard
from exo.download.shard_download import ShardDownloader


class InferenceEngine(ABC):
  session = {}

  @abstractmethod
  async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
    pass

  @abstractmethod
  async def sample(self, x: np.ndarray) -> np.ndarray:
    pass

  @abstractmethod
  async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
    pass

  @abstractmethod
  async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict]]:
    pass

  @abstractmethod
  async def load_checkpoint(self, shard: Shard, path: str):
    pass

  async def save_checkpoint(self, shard: Shard, path: str):
    pass

  async def save_session(self, key, value):
    self.session[key] = value

  async def clear_session(self):
    self.session.empty()

  async def infer_prompt(self, request_id: str, shard: Shard, prompt: str, inference_state: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict]]:
    tokens = await self.encode(shard, prompt)
    # if shard.model_id != 'stable-diffusion-2-1-base':
    #   x = tokens.reshape(1, -1)
    # else:
    #   x = tokens
    output_data, inference_state = await self.infer_tensor(request_id, shard, tokens, inference_state)

    return output_data, inference_state


inference_engine_classes = {
  "mlx": "MLXDynamicShardInferenceEngine",
  "tinygrad": "TinygradDynamicShardInferenceEngine",
  "dummy": "DummyInferenceEngine",
}


def get_inference_engine(inference_engine_name: str, shard_downloader: ShardDownloader):
  if DEBUG >= 2:
    print(f"get_inference_engine called with: {inference_engine_name}")
  if inference_engine_name == "mlx":
    from exo.inference.mlx.sharded_inference_engine import MLXDynamicShardInferenceEngine

    return MLXDynamicShardInferenceEngine(shard_downloader)
  elif inference_engine_name == "tinygrad":
    from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine
    import tinygrad.helpers
    tinygrad.helpers.DEBUG.value = int(os.getenv("TINYGRAD_DEBUG", default="0"))

    return TinygradDynamicShardInferenceEngine(shard_downloader)
  elif inference_engine_name == "dummy":
    from exo.inference.dummy_inference_engine import DummyInferenceEngine
    return DummyInferenceEngine()
  raise ValueError(f"Unsupported inference engine: {inference_engine_name}")
