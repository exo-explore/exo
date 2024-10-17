import asyncio
import numpy as np
from typing import Tuple, Optional
from exo.inference.inference_engine import InferenceEngine
from exo.inference.shard import Shard

class DummyInferenceEngine(InferenceEngine):
    def __init__(self, output_type="static", output_value=None, output_shape=(1,), latency_mean=0.1, latency_stddev=0.01):
        self.output_type = output_type
        self.output_value = output_value
        self.output_shape = output_shape
        self.latency_mean = latency_mean
        self.latency_stddev = latency_stddev
        
        if self.output_type == "static" and self.output_value is None:
            raise ValueError("output_value must be provided when output_type is 'static'.")

    async def simulate_latency(self):
        latency = max(0, np.random.normal(self.latency_mean, self.latency_stddev))
        await asyncio.sleep(latency)

    def generate_output(self):
        if self.output_type == "static":
            return np.array(self.output_value)
        elif self.output_type == "random":
            return np.random.randn(*self.output_shape)

    async def infer_prompt(self, request_id: str, shard: Shard, prompt: str, image_str: Optional[str] = None, inference_state: Optional[str] = None) -> Tuple[np.ndarray, str, bool]:
        await self.simulate_latency()
        output = self.generate_output()
        is_finished = np.random.random() < 0.5  # 50% chance of finishing
        return output, "", is_finished

    async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[str] = None) -> Tuple[np.ndarray, str, bool]:
        await self.simulate_latency()
        output = self.generate_output()
        is_finished = np.random.random() < 0.5 # 50% chance of finishing
        return output, "", is_finished