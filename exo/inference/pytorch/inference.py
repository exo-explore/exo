# experimental, based off of tinygrad/inference.py

import asyncio
from functools import partial
from pathlib import Path
from typing import List, Optional, Union, Callable, Dict
import json
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import os

MODEL_PARAMS = {
    "8B": {
        "args": {
            "dim": 4096,
            "n_heads": 32,
            "n_kv_heads": 8,
            "n_layers": 32,
            "norm_eps": 1e-5,
            "rope_theta": 500000,
            "vocab_size": 128256,
            "hidden_dim": 14336,
        },
        "files": 1,
    },
    "70B": {
        "args": {
            "dim": 8192,
            "n_heads": 64,
            "n_kv_heads": 8,
            "n_layers": 80,
            "norm_eps": 1e-5,
            "rope_theta": 500000,
            "vocab_size": 128256,
            "hidden_dim": 28672,
        },
        "files": 8,
    },
}


# **** helper functions ****
def load(fn: str) -> Union[str, Dict[str, torch.Tensor]]:
    model = ""
    if fn.endswith(".index.json"):
        with open(fn) as fp:
            model = json.load(fp)["model"]

    if model == "":
        model = torch.load(fn, map_location="cpu")

    return model

def build_transformer(model_path: Union[str, Path], model_size="8B", quantize=None, device=None):
    # Load the model configuration and parameters
    model = load(model_path)
    if isinstance(model, str):
        with torch.device(device):
            model = AutoModelForCausalLM.from_pretrained(
                model, 
                torch_dtype=torch.float16 if "cuda" in str(device) else torch.float32, 
                device_map="auto" if "cuda" in str(device) else None
            )
    
    # Quantize the model if specified
    if quantize:
        model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    
    # Shard the model if using multiple devices
    if isinstance(device, tuple):
        for name, param in model.named_parameters():
            if "scale" in name:
                param.data = param.data.chunk(len(device), dim=0)
            elif ".attention." in name:
                param.data = param.data.chunk(len(device), dim=-1)
            elif ".feed_forward.w1." in name or ".feed_forward.w3." in name:
                param.data = param.data.chunk(len(device), dim=0)
            elif ".feed_forward." in name:
                param.data = param.data.chunk(len(device), dim=-1)
            elif "tok_embeddings.weight" in name or "output.weight" in name:
                param.data = param.data.chunk(len(device), dim=0)

    return model

# Sample function using the built transformer
def sample(logits: torch.Tensor, temp: float, k: int, p: float):
    if temp < 1e-6:
        return torch.argmax(logits)

    logits[torch.isnan(logits)] = -float("inf")
    probs = torch.nn.functional.softmax(logits / temp, dim=-1)

    if k:
        top_probs, top_indices = torch.topk(probs, k)
        top_probs = top_probs[top_probs.cumsum(dim=-1) <= p]
        top_indices = top_indices[:len(top_probs)]
        sampled_idx = torch.multinomial(top_probs, 1)
        return top_indices[sampled_idx]

    return torch.multinomial(probs, 1)


# default settings
TEMPERATURE = 0  # 0.85
TOP_K = 25
TOP_P = 0.9
ALPHA_F = 0.1
ALPHA_P = 0.0


def prefill(model, toks, start_pos=0):
    # prefill the model
    for tok in tqdm(toks):
        GlobalCounters.reset()
        inputs = torch.tensor([[tok]], device=model.device)
        model.generate(inputs, do_sample=True, temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P, max_new_tokens=1)
        start_pos += 1
    return start_pos


class PytorchDynamicShardInferenceEngine(InferenceEngine):
    def __init__(self):
        self.shard = None

    async def infer_prompt(self, request_id: str, shard: Shard, prompt: str, image_str: Optional[str] = None, inference_state: Optional[str] = None) -> (np.ndarray, str, bool):
        await self.ensure_shard(shard)
        start_pos = json.loads(inference_state).get("start_pos", 0) if inference_state else 0

        toks = self.tokenizer.encode(prompt)
        start_pos = prefill(self.model, toks[:-1], start_pos=start_pos)
        last_tok = toks[-1]

        input_ids = torch.tensor([[last_tok]], device=self.model.device)
        output_data = self.model.generate(input_ids, do_sample=True, temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P, max_new_tokens=1).cpu().numpy()
        if output_data.size == 1:
            start_pos += 1

        return (
            output_data,
            json.dumps({"start_pos": start_pos}),
            output_data.size == 1 and output_data.item() in [self.tokenizer.eos_token_id],
        )

    async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[str] = None) -> (np.ndarray, str, bool):
        await self.ensure_shard(shard)
        start_pos = json.loads(inference_state).get("start_pos", 0) if inference_state else 0

        input_ids = torch.tensor(input_data, device=self.model.device)
        output_data = self.model.generate(input_ids, do_sample=True, temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P, max_new_tokens=1).cpu().numpy()
        if output_data.size == 1:
            start_pos += 1

        return (
            output_data,
            json.dumps({"start_pos": start_pos}),
            output_data.size == 1 and output_data.item() in [self.tokenizer.eos_token_id],
        )

    async def ensure_shard(self, shard: Shard):
        if self.shard == shard:
            return

        model_path = Path(shard.model_id)
        models_dir = Path(_cache_dir) / "pytorch" / "downloads"
        model_path = models_dir / shard.model_id
        size = "8B"
        if Path(model_path / "tokenizer_config.json").exists():
            model = model_path
        else:
            if DEBUG >= 2:
                print(f"Downloading pytorch model {shard.model_id}...")
            if shard.model_id.lower().find("llama3-8b-sfr") != -1:
                num_files = 4
                for i in range(num_files):
                    await fetch_async(
                        f"https://huggingface.co/mlx-community/Meta-Llama-3-8B-Instruct/resolve/main/model-{(i+1):05d}-of-{num_files:05d}.bin",
                        f"model-{(i+1):05d}-of-{num_files:05d}.bin",
                        subdir=shard.model_id,
                    )
                await fetch_async(
                    "https://huggingface.co/mlx-community/Meta-Llama-3-8B-Instruct/resolve/main/config.json",
                    "config.json",
                    subdir=shard.model_id,
                )
                model = await fetch_async(
                    "https://huggingface.co/mlx-community/Meta-Llama-3-8B-Instruct/raw/main/model.index.json",
                    "model.index.json",
                    subdir=shard.model_id,
                )
                await fetch_async(
                    "https://huggingface.co/mlx-community/Meta-Llama-3-8B-Instruct/resolve/main/special_tokens_map.json",
                    "special_tokens_map.json",
                    subdir=shard.model_id,
                )
                await fetch_async(
                    "https://huggingface.co/mlx-community/Meta-Llama-3-8B-Instruct/resolve/main/tokenizer.json",
                    "tokenizer.json",
                    subdir=shard.model_id,
                )
                await fetch_async(
                    "https://huggingface.co/mlx-community/Meta-Llama-3-8B-Instruct/resolve/main/tokenizer_config.json",
                    "tokenizer_config.json",
                    subdir=shard.model_id,
                )
                size = "8B"
            elif shard.model_id.lower().find("llama3-70b-sfr") != -1:
                raise NotImplementedError("llama3-70b-sfr is not implemented for pytorch")
            else:
                raise ValueError(f"pytorch doesnt currently support arbitrary model downloading. unsupported model: {shard.model_id}")

        model = build_transformer(model_path, shard=shard, model_size=size)
        tokenizer = AutoTokenizer.from_pretrained(model_path if model_path.is_dir() else model_path.parent)

        self.shard = shard
        self.model = model
        self.tokenizer = tokenizer

    def set_on_download_progress(self, on_download_progress: Callable[[int, int], None]):
        pass
