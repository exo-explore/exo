# Adapted from https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/utils.py

import glob
import importlib
import json
import logging
import asyncio
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from huggingface_hub.utils._errors import RepositoryNotFoundError

from mlx_lm.tokenizer_utils import load_tokenizer, TokenizerWrapper
from mlx_lm.tuner.utils import apply_lora_layers

from ..shard import Shard

class ModelNotFoundError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

MODEL_REMAPPING = {
    "mistral": "llama",  # mistral is compatible with llama
    "phi-msft": "phixtral",
}

def _get_classes(config: dict):
    """
    Retrieve the model and model args classes based on the configuration.

    Args:
        config (dict): The model configuration.

    Returns:
        A tuple containing the Model class and the ModelArgs class.
    """
    model_type = config["model_type"]
    model_type = MODEL_REMAPPING.get(model_type, model_type)
    try:
        arch = importlib.import_module(f"exo.inference.mlx.models.{model_type}")
    except ImportError:
        msg = f"Model type {model_type} not supported."
        logging.error(msg)
        raise ValueError(msg)

    return arch.Model, arch.ModelArgs

def load_config(model_path: Path) -> dict:
    try:
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        logging.error(f"Config file not found in {model_path}")
        raise
    return config

def load_model_shard(
    model_path: Path,
    shard: Shard,
    lazy: bool = False,
    model_config: dict = {},
) -> nn.Module:
    """
    Load and initialize the model from a given path.

    Args:
        model_path (Path): The path to load the model from.
        lazy (bool): If False eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``
        model_config(dict, optional): Configuration parameters for the model.
            Defaults to an empty dictionary.

    Returns:
        nn.Module: The loaded and initialized model.

    Raises:
        FileNotFoundError: If the weight files (.safetensors) are not found.
        ValueError: If the model class or args class are not found or cannot be instantiated.
    """

    config = load_config(model_path)
    config.update(model_config)

    # TODO hack
    config["model_type"] = f"sharded_{config['model_type']}"
    config["shard"] = {
        "model_id": model_path.name,
        "start_layer": shard.start_layer,
        "end_layer": shard.end_layer,
        "n_layers": shard.n_layers
    }

    weight_files = glob.glob(str(model_path / "model*.safetensors"))

    if not weight_files:
        # Try weight for back-compat
        weight_files = glob.glob(str(model_path / "weight*.safetensors"))

    if not weight_files:
        logging.error(f"No safetensors found in {model_path}")
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    weights = {}
    all_weights_keys = set()
    for wf in weight_files:
        weights_dict = mx.load(wf)
        all_weights_keys.update(weights_dict.keys())
        weights.update({k: v for k, v in weights_dict.items() if not k.startswith("model.layers.") or shard.start_layer <= int(k.split('.')[2]) <= shard.end_layer})

    model_class, model_args_class = _get_classes(config=config)

    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)

    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    if (quantization := config.get("quantization", None)) is not None:
        # Handle legacy models which may not have everything quantized
        def class_predicate(p, m):
            if not hasattr(m, "to_quantized"):
                return False
            return f"{p}.scales" in all_weights_keys

        nn.quantize(
            model,
            **quantization,
            class_predicate=class_predicate,
        )

    filtered_weights = {}
    for k, v in weights.items():
        if k.startswith("model.layers."):
            layer_num = int(k.split('.')[2])
            if shard.start_layer <= layer_num <= shard.end_layer:
                new_key = f"model.layers.{layer_num - shard.start_layer}." + '.'.join(k.split('.')[3:])
                filtered_weights[new_key] = v
        else:
            filtered_weights[k] = v
    weights = filtered_weights

    model.load_weights(list(weights.items()), strict=False)

    if not lazy:
        mx.eval(model.parameters())

    model.eval()
    return model

async def snapshot_download_async(*args, **kwargs):
    func = partial(snapshot_download, *args, **kwargs)
    return await asyncio.get_event_loop().run_in_executor(None, func)

model_file_to_layers = {
    "mlx-community/Meta-Llama-3-70B-Instruct-4bit": {
        "model-00001-of-00008.safetensors": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "model-00002-of-00008.safetensors": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "model-00003-of-00008.safetensors": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
        "model-00004-of-00008.safetensors": [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42],
        "model-00005-of-00008.safetensors": [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53],
        "model-00006-of-00008.safetensors": [53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64],
        "model-00007-of-00008.safetensors": [64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75],
        "model-00008-of-00008.safetensors": [75, 76, 77, 78, 79]
    }
}

async def get_model_path(path_or_hf_repo: str, shard: Optional[Shard] = None, revision: Optional[str] = None) -> Path:


    """
    Ensures the model is available locally. If the path does not exist locally,
    it is downloaded from the Hugging Face Hub.

    Args:
        path_or_hf_repo (str): The local path or Hugging Face repository ID of the model.
        revision (str, optional): A revision id which can be a branch name, a tag, or a commit hash.

    Returns:
        Path: The path to the model.
    """
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        safetensors_allow_patterns = ["*.safetensors"] if not shard or path_or_hf_repo not in model_file_to_layers else [
            name for name, included_layers in model_file_to_layers[path_or_hf_repo].items()
            if any(layer in range(shard.start_layer, shard.end_layer + 1) for layer in included_layers)
        ]
        print(f"{safetensors_allow_patterns=}")

        try:
            model_path = Path(
                await snapshot_download_async(
                    repo_id=path_or_hf_repo,
                    revision=revision,
                    allow_patterns=[
                        "*.json",
                        "*.py",
                        "tokenizer.model",
                        "*.tiktoken",
                        "*.txt",
                    ] + safetensors_allow_patterns,
                )
            )
        except RepositoryNotFoundError:
            raise ModelNotFoundError(
                f"Model not found for path or HF repo: {path_or_hf_repo}.\n"
                "Please make sure you specified the local path or Hugging Face"
                " repo id correctly.\nIf you are trying to access a private or"
                " gated Hugging Face repo, make sure you are authenticated:\n"
                "https://huggingface.co/docs/huggingface_hub/en/guides/cli#huggingface-cli-login"
            ) from None
    return model_path


async def load_shard(
    path_or_hf_repo: str,
    shard: Shard,
    tokenizer_config={},
    model_config={},
    adapter_path: Optional[str] = None,
    lazy: bool = False,
) -> Tuple[nn.Module, TokenizerWrapper]:
    """
    Load the model and tokenizer from a given path or a huggingface repository.

    Args:
        path_or_hf_repo (Path): The path or the huggingface repository to load the model from.
        tokenizer_config (dict, optional): Configuration parameters specifically for the tokenizer.
            Defaults to an empty dictionary.
        model_config(dict, optional): Configuration parameters specifically for the model.
            Defaults to an empty dictionary.
        adapter_path (str, optional): Path to the LoRA adapters. If provided, applies LoRA layers
            to the model. Default: ``None``.
        lazy (bool): If False eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``
    Returns:
        Tuple[nn.Module, TokenizerWrapper]: A tuple containing the loaded model and tokenizer.

    Raises:
        FileNotFoundError: If config file or safetensors are not found.
        ValueError: If model class or args class are not found.
    """
    model_path = await get_model_path(path_or_hf_repo, shard=shard)

    model = load_model_shard(model_path, shard, lazy, model_config)
    if adapter_path is not None:
        model = apply_lora_layers(model, adapter_path)
        model.eval()
    tokenizer = load_tokenizer(model_path, tokenizer_config)

    return model, tokenizer