# experimental, based off of tinygrad/inference.py
import os
import numpy as np
import asyncio
import json
import torch
from functools import partial
from pathlib import Path
from typing import List, Optional, Union, Callable, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from exo.inference.shard import Shard
from exo.inference.inference_engine import InferenceEngine
from exo.inference.pytorch.helpers import (
    fix_bf16, 
    build_transformer, 
    load_weights, 
    convert_from_huggingface, 
    MODEL_PARAMS
)

# default settings
TEMPERATURE = 0  # 0.85
TOP_K = 25
TOP_P = 0.9
ALPHA_F = 0.1
ALPHA_P = 0.0


# don't think prefill is needed
# think that is used for stats but will look into

class PyTorchDynamicShardInferenceEngine(InferenceEngine):
    def __init__(self):
        self.shard = None
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                temperature=0.7,
                top_k=50,
                top_p=0.95,
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
                temperature=0.7,
                top_k=50,
                top_p=0.95,
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

        cache_dir = Path.home() / ".cache" / "huggingface"
        model_path = cache_dir / "models--" / shard.model_id.replace('/', '--')

        if not model_path.exists():
            print(f"Downloading PyTorch model {shard.model_id}...")
            weights = load_weights(str(model_path / "pytorch_model.bin"))
        else:
            weights = load_weights(str(model_path / "pytorch_model.bin"))

        model_size = "8B"  # Assume 8B model, adjust as needed
        n_heads = MODEL_PARAMS[model_size]["args"]["n_heads"]
        n_kv_heads = MODEL_PARAMS[model_size]["args"]["n_kv_heads"]

        self.model = build_transformer(shard.model_id, device=self.device)
        converted_weights = convert_from_huggingface(weights, self.model, n_heads, n_kv_heads, shard)
        converted_weights = fix_bf16(converted_weights)

        self.model.load_state_dict(converted_weights, strict=False)
        self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.shard = shard

    def set_on_download_progress(self, on_download_progress: Callable[[int, int], None]):
        # This method can be implemented if progress tracking is needed
        pass