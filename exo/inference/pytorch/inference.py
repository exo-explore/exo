# experimental, based off of tinygrad/inference.py
# utilizing pytorch FSDP for sharding

import numpy as np
import json
import torch
from typing import Optional, Callable, Tuple
from transformers import AutoTokenizer
from exo.inference.shard import Shard
from exo.inference.inference_engine import InferenceEngine
from exo.inference.pytorch.helpers import build_transformer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import auto_wrap_policy
from torch.distributed import init_process_group, destroy_process_group

# Default settings
TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 0.95
ALPHA_F = 0.1
ALPHA_P = 0.0

class PyTorchDynamicShardInferenceEngine(InferenceEngine):
    def __init__(self):
        self.shard = None
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize process group
        init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')

    async def infer_prompt(
            self, 
            request_id: str, 
            shard: Shard, 
            prompt: str, 
            image_str: Optional[str] = None, 
            inference_state: Optional[str] = None) -> Tuple[np.ndarray, str, bool]:
        
        await self.ensure_shard(shard)

        start_pos = json.loads(inference_state).get("start_pos", 0) if inference_state else 0

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        attention_mask = torch.ones_like(input_ids)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                do_sample=True,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                top_p=TOP_P,
                pad_token_id=self.tokenizer.eos_token_id,
                start_pos=start_pos
            )

        output_token = outputs[0, -1].item()
        output_data = np.array([output_token])
        start_pos += 1

        is_eos = output_token == self.tokenizer.eos_token_id

        return (
            output_data,
            json.dumps({"start_pos": start_pos}),
            is_eos
        )

    async def infer_tensor(
            self, 
            request_id: str, 
            shard: Shard, 
            input_data: np.ndarray, 
            inference_state: Optional[str] = None) -> Tuple[np.ndarray, str, bool]:
        
        await self.ensure_shard(shard)

        start_pos = json.loads(inference_state).get("start_pos", 0) if inference_state else 0

        input_tensor = torch.tensor(input_data).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_tensor,
                max_new_tokens=1,
                do_sample=True,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                top_p=TOP_P,
                pad_token_id=self.tokenizer.eos_token_id,
                start_pos=start_pos
            )

        output_token = outputs[0, -1].item()
        output_data = np.array([output_token])
        start_pos += 1

        is_eos = output_token == self.tokenizer.eos_token_id

        return (
            output_data,
            json.dumps({"start_pos": start_pos}),
            is_eos
        )

    async def ensure_shard(self, shard: Shard):
        if self.shard == shard:
            return

        # Load model and tokenizer from Hugging Face hub
        self.model = build_transformer(shard.model_id, shard, device=self.device)
        
        # Wrap the model with FSDP
        self.model = FSDP(self.model, auto_wrap_policy=auto_wrap_policy)

        self.tokenizer = AutoTokenizer.from_pretrained(shard.model_id)
        self.shard = shard

    def set_on_download_progress(self, on_download_progress: Callable[[int, int], None]):
        # This method can be implemented if progress tracking is needed
        pass

    def __del__(self):
        destroy_process_group()
