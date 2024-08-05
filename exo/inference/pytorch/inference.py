# experimental, based off of tinygrad/inference.py
# utilizing pytorch FSDP for sharding
# look into shard being optional for the inferece

import numpy as np
import json
import torch
import functools
import os
from typing import Optional, Callable, Tuple
from transformers import AutoTokenizer
from exo.inference.shard import Shard
from exo.inference.inference_engine import InferenceEngine
from exo.inference.pytorch.helpers import build_transformer

# Default settings
TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 0.95
ALPHA_F = 0.1
ALPHA_P = 0.0

class PyTorchDynamicShardInferenceEngine(InferenceEngine):
    def __init__(self, model_name: str = "gpt2", device: str = "cuda", tokenizer: str="gpt2"):
        self.device = device
        self.model_name = model_name
        self.shard = Shard(model_id=model_name, start_layer=0, end_layer=1, n_layers=2)
        self.model = build_transformer(self.shard.model_id, self.shard, device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    async def infer_prompt(
            self, 
            request_id: str, 
            shard: Shard, 
            prompt: str, 
            inference_state: Optional[str] = None) -> Tuple[np.ndarray, str, bool]:
        
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
                pad_token_id=self.tokenizer.eos_token_id
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

    def set_on_download_progress(self, on_download_progress: Callable[[int, int], None]):
        # This method can be implemented if progress tracking is needed
        pass