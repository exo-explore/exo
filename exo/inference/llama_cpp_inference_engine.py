import numpy as np
from exo.inference.inference_engine import InferenceEngine
from exo.inference.shard import Shard
from llama_cpp import Llama

class LlamaCppInferenceEngine(InferenceEngine):
    def __init__(self):
        self.shard = None
        self.model = None
        self.tokenizer = None

    async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
        await self.ensure_shard(shard)
        tokens = self.tokenizer.encode(prompt)
        return np.array(tokens)

    async def sample(self, x: np.ndarray, temp: float = 0.0, top_p: float = 1.0) -> np.ndarray:
        logits = x[:, -1, :]
        if temp == 0:
            token = np.argmax(logits, axis=-1)
        else:
            token = np.random.choice(len(logits), p=np.exp(logits/temp)/np.sum(np.exp(logits/temp)))
        return np.array([token])

    async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
        await self.ensure_shard(shard)
        return self.tokenizer.decode(tokens.tolist())

    async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray) -> np.ndarray:
        await self.ensure_shard(shard)
        output_data = self.model(input_data)
        return np.array(output_data)

    async def load_checkpoint(self, shard: Shard, path: str):
        await self.ensure_shard(shard)
        self.model.load_weights(path)

    async def save_checkpoint(self, shard: Shard, path: str):
        await self.ensure_shard(shard)
        self.model.save_weights(path)

    async def ensure_shard(self, shard: Shard):
        if self.shard == shard:
            return
        self.shard = shard
        self.model = Llama(shard.model_id)
        self.tokenizer = self.model.tokenizer
