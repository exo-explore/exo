from exo.inference.inference_engine import InferenceEngine
import numpy as np
from exo.inference.shard import Shard
from typing import Optional

class OpenVINOInferenceEngine(InferenceEngine):
    async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
        pass

    async def sample(self, x: np.ndarray) -> np.ndarray:
        pass

    async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
        pass

    async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict]]:
        pass

    async def load_checkpoint(self, shard: Shard, path: str):
        pass
