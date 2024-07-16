import numpy as np

from typing import Tuple, Optional
from abc import ABC, abstractmethod
from .shard import Shard

class InferenceEngine(ABC):
    @abstractmethod
    async def infer_tensor(self, shard: Shard, input_data: np.ndarray, inference_state: Optional[str] = None) -> Tuple[np.ndarray, str, bool]:
        pass

    @abstractmethod
    async def infer_prompt(self, shard: Shard, prompt: str, inference_state: Optional[str] = None) -> Tuple[np.ndarray, str, bool]:
        pass

    @abstractmethod
    async def reset_shard(self, shard: Shard):
        pass
