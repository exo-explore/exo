# Adapted from https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/utils.py

import glob
import importlib
import json
import logging
import asyncio
import aiohttp
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, Union, List, Callable
from PIL import Image
from io import BytesIO
import base64

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download, list_repo_tree, get_paths_info
from huggingface_hub.utils import filter_repo_objects
from huggingface_hub.file_download import repo_folder_name
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.utils._errors import RepositoryNotFoundError
from transformers import AutoProcessor

from mlx_lm.tokenizer_utils import load_tokenizer, TokenizerWrapper
from mlx_lm.tuner.utils import apply_lora_layers

from exo import DEBUG
from exo.inference.hf_helpers import download_all_files, HFRepoProgressCallback
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
  config["shard"] = {
    "model_id": model_path.name,
    "start_layer": shard.start_layer,
    "end_layer": shard.end_layer,
    "n_layers": shard.n_layers,
  }

  weight_files = glob.glob(str(model_path / "model*.safetensors"))

  if not weight_files:
    # Try weight for back-compat
    weight_files = glob.glob(str(model_path / "weight*.safetensors"))

  if not weight_files:
    logging.error(f"No safetensors found in {model_path}")
    raise FileNotFoundError(f"No safetensors found in {model_path}")

  weights = {}
  for wf in sorted(weight_files):
    if DEBUG >= 8:
      layer_nums = set()
      for k in mx.load(wf):
        if k.startswith("model.layers."):
          layer_num = int(k.split(".")[2])
          layer_nums.add(layer_num)
        if k.startswith("language_model.model.layers."):
          layer_num = int(k.split(".")[3])
          layer_nums.add(layer_num)
      print(f"\"{wf.split('/')[-1]}\": {sorted(layer_nums)},")

    weights.update(mx.load(wf))

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
        return f"{p}.scales" in weights
    nn.quantize(
      model,
      **quantization,
      class_predicate=class_predicate,
    )

  model.load_weights(list(weights.items()), strict=True)

  if not lazy:
    mx.eval(model.parameters())

  model.eval()
  return model


repo_id_safetensors_layers = {
  "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit": {
    "model.safetensors": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
  },
  "mlx-community/Meta-Llama-3.1-70B-Instruct-4bit": {
    "model-00001-of-00008.safetensors": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "model-00002-of-00008.safetensors": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    "model-00003-of-00008.safetensors": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
    "model-00004-of-00008.safetensors": [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42],
    "model-00005-of-00008.safetensors": [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53],
    "model-00006-of-00008.safetensors": [53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64],
    "model-00007-of-00008.safetensors": [64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75],
    "model-00008-of-00008.safetensors": [75, 76, 77, 78, 79],
  },
  "mlx-community/Meta-Llama-3.1-405B-Instruct-4bit": {
    "model-00001-of-00046.safetensors": [0, 1, 2],
    "model-00002-of-00046.safetensors": [2, 3, 4, 5],
    "model-00003-of-00046.safetensors": [5, 6, 7],
    "model-00004-of-00046.safetensors": [8, 9, 10],
    "model-00005-of-00046.safetensors": [10, 11, 12, 13],
    "model-00006-of-00046.safetensors": [13, 14, 15, 16],
    "model-00007-of-00046.safetensors": [16, 17, 18, 19],
    "model-00008-of-00046.safetensors": [19, 20, 21],
    "model-00009-of-00046.safetensors": [22, 23, 24],
    "model-00010-of-00046.safetensors": [24, 25, 26, 27],
    "model-00011-of-00046.safetensors": [27, 28, 29, 30],
    "model-00012-of-00046.safetensors": [30, 31, 32, 33],
    "model-00013-of-00046.safetensors": [33, 34, 35],
    "model-00014-of-00046.safetensors": [36, 37, 38],
    "model-00015-of-00046.safetensors": [38, 39, 40, 41],
    "model-00016-of-00046.safetensors": [41, 42, 43, 44],
    "model-00017-of-00046.safetensors": [44, 45, 46, 47],
    "model-00018-of-00046.safetensors": [47, 48, 49],
    "model-00019-of-00046.safetensors": [50, 51, 52],
    "model-00020-of-00046.safetensors": [52, 53, 54, 55],
    "model-00021-of-00046.safetensors": [55, 56, 57, 58],
    "model-00022-of-00046.safetensors": [58, 59, 60, 61],
    "model-00023-of-00046.safetensors": [61, 62, 63],
    "model-00024-of-00046.safetensors": [64, 65, 66],
    "model-00025-of-00046.safetensors": [66, 67, 68, 69],
    "model-00026-of-00046.safetensors": [69, 70, 71, 72],
    "model-00027-of-00046.safetensors": [72, 73, 74, 75],
    "model-00028-of-00046.safetensors": [75, 76, 77],
    "model-00029-of-00046.safetensors": [78, 79, 80],
    "model-00030-of-00046.safetensors": [80, 81, 82, 83],
    "model-00031-of-00046.safetensors": [83, 84, 85, 86],
    "model-00032-of-00046.safetensors": [86, 87, 88, 89],
    "model-00033-of-00046.safetensors": [89, 90, 91],
    "model-00034-of-00046.safetensors": [92, 93, 94],
    "model-00035-of-00046.safetensors": [94, 95, 96, 97],
    "model-00036-of-00046.safetensors": [97, 98, 99, 100],
    "model-00037-of-00046.safetensors": [100, 101, 102, 103],
    "model-00038-of-00046.safetensors": [103, 104, 105],
    "model-00039-of-00046.safetensors": [106, 107, 108],
    "model-00040-of-00046.safetensors": [108, 109, 110, 111],
    "model-00041-of-00046.safetensors": [111, 112, 113, 114],
    "model-00042-of-00046.safetensors": [114, 115, 116, 117],
    "model-00043-of-00046.safetensors": [117, 118, 119],
    "model-00044-of-00046.safetensors": [120, 121, 122],
    "model-00045-of-00046.safetensors": [122, 123, 124, 125],
    "model-00046-of-00046.safetensors": [125]
  },
  "mlx-community/Mistral-Nemo-Instruct-2407-4bit": {
    "model-00001-of-00002.safetensors": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
    "model-00002-of-00002.safetensors": [32, 33, 34, 35, 36, 37, 38, 39],
  },
  "mlx-community/Mistral-Large-Instruct-2407-4bit": {
    "model-00001-of-00014.safetensors": [0, 1, 2, 3, 4, 5, 6],
    "model-00002-of-00014.safetensors": [6, 7, 8, 9, 10, 11, 12, 13],
    "model-00003-of-00014.safetensors": [13, 14, 15, 16, 17, 18, 19, 20],
    "model-00004-of-00014.safetensors": [20, 21, 22, 23, 24, 25, 26],
    "model-00005-of-00014.safetensors": [27, 28, 29, 30, 31, 32, 33],
    "model-00006-of-00014.safetensors": [33, 34, 35, 36, 37, 38, 39, 40],
    "model-00007-of-00014.safetensors": [40, 41, 42, 43, 44, 45, 46, 47],
    "model-00008-of-00014.safetensors": [47, 48, 49, 50, 51, 52, 53, 54],
    "model-00009-of-00014.safetensors": [54, 55, 56, 57, 58, 59, 60],
    "model-00010-of-00014.safetensors": [61, 62, 63, 64, 65, 66, 67],
    "model-00011-of-00014.safetensors": [67, 68, 69, 70, 71, 72, 73, 74],
    "model-00012-of-00014.safetensors": [74, 75, 76, 77, 78, 79, 80, 81],
    "model-00013-of-00014.safetensors": [81, 82, 83, 84, 85, 86, 87],
    "model-00014-of-00014.safetensors": [87]
  },
  "llava-hf/llava-1.5-7b-hf": {
    "model-00001-of-00003.safetensors": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "model-00002-of-00003.safetensors": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
    "model-00003-of-00003.safetensors": [22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
  }
}

def get_safetensors_allow_patterns(repo_id: str, shard: Optional[Shard] = None):
    return ["*.safetensors"] # TODO: enable this
    if not shard:
      return ["*.safetensors"]

    allow_patterns = []
    for repo_id, safetensors_layers in repo_id_safetensors_layers.items():
        if repo_id == shard.model_id:
            for safetensor, layers in safetensors_layers.items():
                if any(shard.start_layer <= layer <= shard.end_layer for layer in layers):
                    allow_patterns.append(safetensor)

    return allow_patterns if len(allow_patterns) > 0 else ["*.safetensors"]

async def get_model_path(path_or_hf_repo: str, shard: Optional[Shard] = None, revision: str = "main", progress_callback: Optional[HFRepoProgressCallback] = None) -> Path:
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
    try:
      model_path = Path(
        await download_all_files(
          repo_id=path_or_hf_repo,
          revision=revision,
          allow_patterns=[
            "*.json",
            "*.py",
            "tokenizer.model",
            "*.tiktoken",
            "*.txt",
          ] + get_safetensors_allow_patterns(path_or_hf_repo, shard),
          progress_callback=progress_callback,
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
  progress_callback: Optional[HFRepoProgressCallback] = None,
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
  model_path = await get_model_path(path_or_hf_repo, shard, progress_callback=progress_callback)

  model = load_model_shard(model_path, shard, lazy, model_config)
  if adapter_path is not None:
    model = apply_lora_layers(model, adapter_path)
    model.eval()

  # TODO: figure out a generic solution
  if model.model_type == "llava":
    processor = AutoProcessor.from_pretrained(model_path)
    processor.eos_token_id = processor.tokenizer.eos_token_id
    processor.encode = processor.tokenizer.encode
    return model, processor
  else:
    tokenizer = load_tokenizer(model_path, tokenizer_config)
    return model, tokenizer

async def get_image_from_str(_image_str: str):
    image_str = _image_str.strip()

    if image_str.startswith("http"):
        async with aiohttp.ClientSession() as session:
            async with session.get(image_str, timeout=10) as response:
                content = await response.read()
                return Image.open(BytesIO(content)).convert("RGB")
    elif image_str.startswith("data:image/"):
        # Extract the image format and base64 data
        format_prefix, base64_data = image_str.split(";base64,")
        image_format = format_prefix.split("/")[1].lower()
        if DEBUG >= 2: print(f"{image_str=} {image_format=}")
        imgdata = base64.b64decode(base64_data)
        img = Image.open(BytesIO(imgdata))

        # Convert to RGB if not already
        if img.mode != "RGB":
            img = img.convert("RGB")

        return img
    else:
        raise ValueError("Invalid image_str format. Must be a URL or a base64 encoded image.")
