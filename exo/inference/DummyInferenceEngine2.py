import json
import numpy as np
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from exo.inference.inference_engine import InferenceEngine
import asyncio
import time

class DummyInferenceEngine2(InferenceEngine):
    def __init__(self, shard_downloader):
        self.shard = None
        self.shard_downloader = shard_downloader
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.latency_mean = 0.1
        self.latency_stddev = 0.1

    def _dummy_tokenize(self, text: str) -> np.ndarray:
        """Simulate tokenization in a CPU-bound manner"""
        # Simulate some CPU work
        result = []
        for c in text:
            # Add some arbitrary computation to simulate work
            val = ord(c)
            for _ in range(1000):  # Simulate processing
                val = (val * 31 + 7) % 256
            result.append(val)
        return np.array([result])

    def _dummy_inference(self, tokens: np.ndarray, start_pos: int) -> np.ndarray:
        """Simulate model inference in a CPU-bound manner"""
        # Simulate some CPU work
        result = []
        for token in tokens.flatten():
            # Add some arbitrary computation to simulate work
            val = int(token)
            for _ in range(1000):  # Simulate processing
                val = (val * 17 + 5) % 256
            result.append(val)
        
        # Simulate random latency
        time.sleep(max(0, np.random.normal(self.latency_mean, self.latency_stddev)))
        return np.array(result)

    async def infer_prompt(self, request_id: str, shard, prompt: str, image_str: Optional[str] = None, inference_state: Optional[str] = None) -> Tuple[np.ndarray, str, bool]:
        await self.ensure_shard(shard)
        # Parse state
        state = json.loads(inference_state or "{}")
        start_pos = state.get("start_pos", 0)
        n_captured_toks = state.get("n_captured_toks", 0)

        # Run tokenization in thread pool
        toks = await asyncio.get_event_loop().run_in_executor(self.executor, self._dummy_tokenize,prompt)

        # Run inference in thread pool
        output = await asyncio.get_event_loop().run_in_executor(self.executor, self._dummy_inference, toks, start_pos   )

        # Update state
        start_pos += len(prompt)
        n_captured_toks = len(prompt)
        
        # Randomly decide if we're finished (20% chance)
        is_finished = np.random.random() < 0.2

        if is_finished:
            start_pos += 1
            n_captured_toks = 0
            output = np.array([[output[0]]])  # Return single token for completion

        return output, json.dumps({
            "start_pos": start_pos,
            "n_captured_toks": n_captured_toks
        }), is_finished

    async def infer_tensor(self, request_id: str, shard, input_data: np.ndarray, inference_state: Optional[str] = None) -> Tuple[np.ndarray, str, bool]:
        await self.ensure_shard(shard)
        
        # Parse state
        state = json.loads(inference_state or "{}")
        start_pos = state.get("start_pos", 0)
        n_captured_toks = state.get("n_captured_toks", 0)

        # Run inference in thread pool
        output = await asyncio.get_event_loop().run_in_executor(self.executor, self._dummy_inference, input_data, start_pos)

        # Update state
        start_pos += input_data.shape[1]
        n_captured_toks = input_data.shape[1]
        
        # Randomly decide if we're finished (20% chance)
        is_finished = np.random.random() < 0.2

        if is_finished:
            start_pos += 1
            n_captured_toks = 0
            output = np.array([[output[0]]])  # Return single token for completion

        return output, json.dumps({
            "start_pos": start_pos,
            "n_captured_toks": n_captured_toks
        }), is_finished

    async def ensure_shard(self, shard):
        # Do nothing, as we're not actually loading any model
        if self.shard != shard:
            # Simulate shard loading time
            await asyncio.sleep(0.1)
            self.shard = shard