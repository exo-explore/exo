import numpy as np
import mlx.nn as nn

from abc import ABC, abstractmethod
from .shard import Shard

class InferenceEngine(ABC):
    @abstractmethod
    async def infer_shard(self, shard: Shard, input_data: np.ndarray) -> np.ndarray:
        pass

    async def infer_prompt(self, shard: Shard, prompt: str) -> np.ndarray:
        pass

    @abstractmethod
    async def reset_shard(self, shard: Shard):
        pass
