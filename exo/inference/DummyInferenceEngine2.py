import json
import numpy as np
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from exo.inference.inference_engine import InferenceEngine

class DummyInferenceEngine2(InferenceEngine):
    def __init__(self, shard_downloader):
        self.shard = None
        self.shard_downloader = shard_downloader
        self.executor = ThreadPoolExecutor(max_workers=1)

    async def infer_prompt(self, request_id: str, shard, prompt: str, image_str: Optional[str] = None, inference_state: Optional[str] = None) -> Tuple[np.ndarray, str, bool]:
        # Dummy tokenization (just convert to ASCII values)
        toks = np.array([[ord(c) for c in prompt]])
        
        # Create dummy state
        state = json.loads(inference_state or "{}")
        state["start_pos"] = state.get("start_pos", 0) + len(prompt)
        state["n_captured_toks"] = len(prompt)

        # Return input as output
        return toks, json.dumps(state), False

    async def infer_tensor(self, request_id: str, shard, input_data: np.ndarray, inference_state: Optional[str] = None) -> Tuple[np.ndarray, str, bool]:
        # Create dummy state
        state = json.loads(inference_state or "{}")
        state["start_pos"] = state.get("start_pos", 0) + input_data.shape[1]
        state["n_captured_toks"] = input_data.shape[1]

        # Return input as output
        return input_data, json.dumps(state), False

    async def ensure_shard(self, shard):
        # Do nothing, as we're not actually loading any model
        self.shard = shard