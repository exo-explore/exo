import numpy as np
import mlx.nn as nn

from abc import ABC, abstractmethod
from .shard import Shard

class InferenceEngine(ABC):
    @abstractmethod
    async def infer_shard(self, shard: Shard, input_data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    async def reset_shard(self, shard: Shard):
        pass

class MLXFixedShardInferenceEngine(InferenceEngine):
    def __init__(self, model: nn.Module, shard: Shard):
        self.model = model
        self.shard = shard

    async def infer_shard(self, shard: Shard, input_data: np.ndarray) -> np.ndarray:
        if shard != self.shard:
            raise ValueError(f"Shard mismatch: {shard} != {self.shard}")

        output_data = self.model.process(input_data)
        print("Processed data through model shard")
        return output_data

    async def reset_shard(self, shard: Shard):
        # TODO
        print(f"Resetting shard: {shard}")