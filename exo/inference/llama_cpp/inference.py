import numpy as np
from typing import Optional, Tuple
from llama_cpp import Llama
from exo.inference.shard import Shard
from exo.inference.inference_engine import InferenceEngine
from exo.download.shard_download import ShardDownloader

class LlamaCppDynamicShardEngine(InferenceEngine):
    def __init__(self, shard_downloader: ShardDownloader):
        self.shard = None
        self.shard_downloader = shard_downloader

    async def infer_prompt(self, request_id: str, shard: Shard, prompt: str, image_str: Optional[str] = None, inference_state: Optional[str] = None) -> (np.ndarray, str, bool):
        await self.ensure_shard(shard)

    async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[str] = None) -> Tuple[np.ndarray, str, bool]:
        await self.ensure_shard(shard)

    async def ensure_shard(self, shard: Shard):
        if self.shard == shard:
            return

        model_path = await self.shard_downloader.ensure_shard(shard)
        self.shard = shard
