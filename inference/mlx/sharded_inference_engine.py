import numpy as np
import mlx.core as mx
from ..inference_engine import InferenceEngine
from .sharded_model import StatefulShardedModel
from .sharded_utils import load_shard
from ..shard import Shard

class MLXFixedShardInferenceEngine(InferenceEngine):
    def __init__(self, model_path: str, shard: Shard):
        print("initializing fixed shard inference", shard)
        self.shard = shard
        model_shard, self.tokenizer = load_shard(model_path, shard)
        self.stateful_sharded_model = StatefulShardedModel(shard, model_shard)

    async def infer_prompt(self, shard: Shard, prompt: str) -> np.ndarray:
        if shard != self.shard:
            raise ValueError(f"Shard mismatch: {shard} != {self.shard}")

        output_data = self.stateful_sharded_model.step(mx.array(self.tokenizer.encode(prompt)))
        return np.array(output_data)

    async def infer_shard(self, shard: Shard, input_data: np.ndarray) -> np.ndarray:
        if shard != self.shard:
            raise ValueError(f"Shard mismatch: {shard} != {self.shard}")

        print("infer_shard", shard, input_data)

        output_data = self.stateful_sharded_model.step(mx.array(input_data))
        return np.array(output_data)

    async def reset_shard(self, shard: Shard):
        if shard != self.shard:
            raise ValueError(f"Shard mismatch: {shard} != {self.shard}")

        print(f"Resetting shard: {shard}")
        self.stateful_sharded_model.reset()
