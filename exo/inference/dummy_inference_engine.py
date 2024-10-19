from typing import Optional, Tuple, TYPE_CHECKING
import numpy as np
import asyncio
import json
from typing import Optional, Tuple
if TYPE_CHECKING:
    from exo.inference.inference_engine import InferenceEngine
from exo.inference.inference_engine import InferenceEngine
from exo.inference.shard import Shard

class DummyInferenceEngine(InferenceEngine):
    def __init__(self, shard_downloader):
        self.shard = None
        self.shard_downloader = shard_downloader
        self.vocab_size = 1000
        self.eos_token_id = 0
        self.latency_mean = 0.1
        self.latency_stddev = 0.02

    async def infer_prompt(self, request_id: str, shard: Shard, prompt: str, image_str: Optional[str] = None, inference_state: Optional[str] = None) -> Tuple[np.ndarray, str, bool]:
        try:
            await self.ensure_shard(shard)
            
            # Generate random tokens
            output_length = np.random.randint(1, 10)
            output = np.random.randint(1, self.vocab_size, size=(1, output_length))
            
            # Simulate latency
            await asyncio.sleep(max(0, np.random.normal(self.latency_mean, self.latency_stddev)))
            
            # Randomly decide if finished
            is_finished = np.random.random() < 0.2
            if is_finished:
                output = np.array([[self.eos_token_id]])
            
            new_state = json.dumps({"dummy_state": "some_value"})
            
            return output, new_state, is_finished
        except Exception as e:
            print(f"Error in DummyInferenceEngine.infer_prompt: {str(e)}")
            return np.array([[self.eos_token_id]]), json.dumps({"error": str(e)}), True

    async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[str] = None) -> Tuple[np.ndarray, str, bool]:
        await self.ensure_shard(shard)
        state = json.loads(inference_state or "{}")
        start_pos = state.get("start_pos", 0)
        
        output_length = np.random.randint(1, 10)
        output = np.random.randint(1, self.vocab_size, size=(1, output_length))
        
        await asyncio.sleep(max(0, np.random.normal(self.latency_mean, self.latency_stddev)))
        
        is_finished = np.random.random() < 0.2
        if is_finished:
            output = np.array([[self.eos_token_id]])
        
        start_pos += input_data.shape[1] + output_length
        new_state = json.dumps({"start_pos": start_pos})
        
        return output, new_state, is_finished

    async def ensure_shard(self, shard: Shard):
        if self.shard == shard:
            return
        # Simulate shard loading without making any API calls
        await asyncio.sleep(0.1)  # Simulate a short delay
        self.shard = shard
        print(f"DummyInferenceEngine: Simulated loading of shard {shard.model_id}")