from exo.inference.inference_engine import InferenceEngine
from transformers import AutoModelForCausalLM, AutoTokenizer
from exo.inference.shard import Shard
from exo.download.shard_download import ShardDownloader

def download_model_from_model_id(model_id):
    pass

class HuggingfaceInferenceEngine(InferenceEngine):
    def __init__(self, shard_downloader: ShardDownloader):
        self.shard_downloader = shard_downloader
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
        """ Loads the wieght into the model defined after enuring shard exits"""
        self.ensure_shard(shard)
        return 
    
    async def ensure_shard(self, shard: Shard):
        """make hure the ard exists otherwise downloads the model and the tokenizer and initialise the shard """
        if self.shard == shard:
            return
        
        model_path = await self.shard_downloader.ensure_shard(shard, self.__class__.__name__)
        if self.shard != shard:
            self.shard = shard
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
