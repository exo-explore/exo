from exo.inference.shard import Shard
from exo.inference.inference_engine import InferenceEngine
from exo.download.shard_download import ShardDownloader
from typing import Tuple, Optional
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class LLamaInferenceEngine(InferenceEngine):
    def __init__(self, shard_downloader: ShardDownloader):
        self.shard_downloader = shard_downloader
        self.executor = ThreadPoolExecutor(max_workers=1)

    def infer_prompt(self, request_id: str, shard: Shard, prompt: str, image_str: Optional[str] = None, inference_state: Optional[str] = None) -> (np.ndarray, str, bool):
        pass

    def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[str] = None) -> Tuple[np.ndarray, str, bool]:
        pass
