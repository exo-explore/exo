from pathlib import Path
import json
import os
from exo.inference.inference_engine import InferenceEngine
from exo.download.shard_download import ShardDownloader
from concurrent.futures import ThreadPoolExecutor
import DummyModel
import asyncio
from exo.inference.shard import Shard


# default settings
TEMPERATURE = 0.85
TOP_K = 25
TOP_P = 0.9
ALPHA_F = 0.1
ALPHA_P = 0.0


class DummyShardInferenceEngine(InferenceEngine):
    def __init__(self, shard_downloader: ShardDownloader):
        self.shard = None
        self.shard_downloader = shard_downloader
        self.executor = ThreadPoolExecutor(max_workers=1)
        
        # Initialize dummy model with parameters from MODEL_PARAMS
        self.model = DummyModel(dim=4096, n_heads=32)  # Example parameters for "8B" model

    async def infer_prompt(self, request_id: str, shard: Shard, prompt: str, image_str: Optional[str] = None, inference_state: Optional[str] = None) -> (np.ndarray, str, bool):
        await self.ensure_shard(shard)
        start_pos = json.loads(inference_state or "{}").get("start_pos", 0)
        n_captured_toks = json.loads(inference_state or "{}").get("n_captured_toks", 0)

        toks = await asyncio.get_event_loop().run_in_executor(self.executor, self.tokenizer.encode, prompt)
        
        # Use the dummy model for inference
        h = await asyncio.get_event_loop().run_in_executor(self.executor,
            lambda: self.model(Tensor([toks]), start_pos, TEMPERATURE))

        if h.shape == (1,):
            start_pos += len(toks) + 1
            n_captured_toks = 0
            return np.array([[random.randint(0, 127)]], dtype=np.int32), json.dumps({"start_pos": start_pos, "n_captured_toks": n_captured_toks}), True
        else:
            n_captured_toks = len(toks)
            return h.numpy(), json.dumps({"start_pos": start_pos, "n_captured_toks": n_captured_toks}), False

    async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[str] = None) -> Tuple[np.ndarray, str, bool]:
        await self.ensure_shard(shard)
        start_pos = json.loads(inference_state or "{}").get("start_pos", 0)
        n_captured_toks = json.loads(inference_state or "{}").get("n_captured_toks", 0)

        # Use the dummy model for inference
        h = await asyncio.get_event_loop().run_in_executor(self.executor,
            lambda: self.model(Tensor(input_data), start_pos, TEMPERATURE))

        if h.shape == (1,):
            start_pos += n_captured_toks + 1
            n_captured_toks = 0
            return np.array([[random.randint(0, 127)]], dtype=np.int32), json.dumps({"start_pos": start_pos, "n_captured_toks": n_captured_toks}), True
        else:
            return h.numpy(), json.dumps({"start_pos": start_pos, "n_captured_toks": n_captured_toks}), False

    async def ensure_shard(self, shard: Shard):
        # This method can remain unchanged or be modified as needed.