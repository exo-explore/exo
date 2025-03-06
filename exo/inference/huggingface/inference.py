from exo.inference.inference_engine import InferenceEngine


class HuggingfaceInferenceEngine(InferenceEngine):
    def __init__(self):
        pass
    
    async def encode(self, shard, prompt):
        pass
    
    async def sample(self, x):
        pass
    
    async def decode(self, shard, tokens):
        pass
    
    async def infer_tensor(self, request_id, shard, input_data, inference_state=None):
        pass
    
    async def load_checkpoint(self, shard, path):
        pass
    
    async def ensure_shard(self, shard):
        pass
    
    async def infer_prompt(self, request_id, shard, prompt, inference_state=None):
        pass