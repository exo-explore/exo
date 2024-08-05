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
def concat_weights(models, device=None):
    def convert(name) -> torch.Tensor:
        disk_tensors: List[torch.Tensor] = [model[name] for model in models]
        if len(disk_tensors) == 1 or len(disk_tensors[0].shape) == 1:
            return disk_tensors[0].to(device=device)
        axis = 1 if name.endswith(".attention.wo.weight") or name.endswith(".feed_forward.w2.weight") else 0
        lazy_tensors = [data.to(device=device) for data in disk_tensors]
        return torch.cat(lazy_tensors, dim=axis)

    return {name: convert(name) for name in {name for model in models for name in model}}

def load(fn: str) -> Union[str, Dict[str, torch.Tensor]]:
    model = ""
    if fn.endswith("pytorch_model.bin.index.json"):
        with open(fn) as fp:
            weight_map = json.load(fp)["weight_map"]
        parts = {n: torch.load(str(Path(fn).parent / Path(n).name), map_location="cpu") for n in set(weight_map.values())}
        return {k: parts[n][k] for k, n in weight_map.items()}
    else:
        model = torch.load(fn, map_location="cpu")
    return model

def build_transformer(model_name: str, model_size="8B", quantize=None, device=None):
    with torch.device(device):
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
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

# default settings
TEMPERATURE = 0  # 0.85
TOP_K = 25
TOP_P = 0.9
ALPHA_F = 0.1
ALPHA_P = 0.0

