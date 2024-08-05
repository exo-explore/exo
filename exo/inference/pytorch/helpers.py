# Helper functions for pytorch inference
# Some code coming from tinygrad but written towards pytorch

# import os
# import numpy as np
# import asyncio
import json
import torch
# from functools import partial
from pathlib import Path
from typing import List,  Union, Dict, Any
from transformers import AutoModelForCausalLM
from exo.inference.shard import Shard
# from exo.inference.inference_engine import InferenceEngine

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

def concat_weights(models, device=None):
    """
    Concatenates weights from multiple model parts along the appropriate axis.

    Args:
        models (List[Dict[str, torch.Tensor]]): List of dictionaries containing model weights.
        device (Optional[torch.device]): The device to move the weights to (e.g., 'cpu' or 'cuda').

    Returns:
        Dict[str, torch.Tensor]: A dictionary where the keys are the weight names and the values 
        are the concatenated tensors moved to the specified device.
    """
    def convert(name) -> torch.Tensor:
        disk_tensors: List[torch.Tensor] = [model[name] for model in models]
        if len(disk_tensors) == 1 or len(disk_tensors[0].shape) == 1:
            return disk_tensors[0].to(device=device)
        
        ewn = name.endswith(".attention.wo.weight") or name.endswith(".feed_forward.w2.weight")
        axis = 1 if ewn else 0

        lazy_tensors = [data.to(device=device) for data in disk_tensors]
        return torch.cat(lazy_tensors, dim=axis)

    return {name: convert(name) for name in {name for model in models for name in model}}

def load_weights(fn: str) -> Union[str, Dict[str, torch.Tensor]]:
    """
    Loads model weights from a specified file. Supports both individual model files and 
    index files that map to multiple weight files.

    Args:
        fn (str): The file path to load weights from.

    Returns:
        Union[str, Dict[str, torch.Tensor]]: A string representing the model or a 
        dictionary of model weights.
    """
    model = ""
    if fn.endswith("pytorch_model.bin.index.json"):
        with open(fn) as fp:
            weight_map = json.load(fp)["weight_map"]
        
        for n in set(weight_map.values()):
            full_path = str(Path(fn).parent / Path(n).name)
            parts = {n: torch.load(full_path, map_location="cpu")}

        return {k: parts[n][k] for k, n in weight_map.items()}
    else:
        model = torch.load(fn, map_location="cpu")
    return model

def convert_from_huggingface(
        weights: Dict[str, torch.Tensor], 
        model: torch.nn.Module, 
        n_heads: int, 
        n_kv_heads: int, 
        shard: Shard) -> Dict[str, torch.Tensor]:
    """
    Converts Hugging Face model weights to the format expected by the target model.

    Args:
        weights (Dict[str, torch.Tensor]): Dictionary of Hugging Face model weights.
        model (nn.Module): The target model.
        n_heads (int): Number of attention heads.
        n_kv_heads (int): Number of key-value heads.
        shard (Shard): Shard object containing information about the model shard.

    Returns:
        Dict[str, torch.Tensor]: Dictionary of converted weights.
    """
    def permute(v: torch.Tensor, n_heads: int) -> torch.Tensor:
        return v.view(
            n_heads, 
            2, 
            v.shape[0] // (2 * n_heads), 
            v.shape[1]
        ).transpose(1, 2).reshape(*v.shape)

    keymap = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        **{f"model.layers.{l}.input_layernorm.weight": f"layers.{l}.attention_norm.weight" for l in range(len(model.layers))},
        **{f"model.layers.{l}.self_attn.{x}_proj.weight": f"layers.{l}.attention.w_{x}.weight" for x in ["q", "k", "v", "o"] for l in range(len(model.layers))},
        **{f"model.layers.{l}.post_attention_layernorm.weight": f"layers.{l}.ffn_norm.weight" for l in range(len(model.layers))},
        **{f"model.layers.{l}.mlp.{x}_proj.weight": f"layers.{l}.feed_forward.w_{y}.weight" for x, y in {"gate": "1", "down": "2", "up": "3"}.items() for l in range(len(model.layers))},
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight",
    }

    sd = {}
    for k, v in weights.items():
        if ".rotary_emb." in k:
            continue

        if "model.layers" in k:
            layer_num = int(k.split(".")[2])
            if shard.start_layer <= layer_num <= shard.end_layer:
                k = f"model.layers.{layer_num - shard.start_layer}." + ".".join(k.split(".")[3:])
            else:
                continue

            if "q_proj" in k:
                v = permute(v, n_heads)
            elif "k_proj" in k:
                v = permute(v, n_kv_heads)
                
        if k in keymap:
            sd[keymap[k]] = v

    return sd

def fix_bf16(weights: Dict[Any, torch.Tensor]) -> Dict[Any, torch.Tensor]:
    """
    Converts weights to bfloat16 if supported by the device, otherwise to float16.

    Args:
        weights (Dict[Any, torch.Tensor]): Dictionary of model weights.

    Returns:
        Dict[Any, torch.Tensor]: Dictionary of converted weights.
    """
    supports_bf16 = torch.cuda.is_bf16_supported()

    if supports_bf16:
        return {k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v for k, v in weights.items()}
    else:
        return {k: v.to(torch.float16) if v.dtype == torch.bfloat16 else v for k, v in weights.items()}


def build_transformer(model_name: str, shard: Shard, model_size="8B", quantize=None, device=None):
    """
    Builds a transformer model by loading it from the Hugging Face model hub and applying 
    weight conversion, quantization, and sharding as specified.

    Args:
        model_name (str): The name of the model to load from the Hugging Face model hub.
        shard (Shard): A Shard object containing information about the model shard.
        model_size (str, optional): The size of the model to load (default is "8B").
        quantize (bool, optional): Whether to apply dynamic quantization to the model (default is None).
        device (torch.device, optional): The device to load the model onto (default is None).

    Returns:
        nn.Module: The constructed and configured transformer model.
    """
    # Load model from Hugging Face hub
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if "cuda" in str(device) else torch.float32,
        device_map="auto" if "cuda" in str(device) else None
    )

    # Quantize the model if specified
    if quantize:
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear})

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

