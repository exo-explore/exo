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
import traceback

import mlx.core as mx
import mlx.nn as nn
from transformers import AutoProcessor

from mlx_lm.tokenizer_utils import load_tokenizer, TokenizerWrapper

from exo import DEBUG
from exo.inference.tokenizers import resolve_tokenizer
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
    traceback.print_exc()
    raise ValueError(msg)

  return arch.Model, arch.ModelArgs


def load_config(model_path: Path) -> dict:
  try:
    config_path = model_path / "config.json"
    if config_path.exists():
      with open(config_path, "r") as f:
        config = json.load(f)
      return config
    
    model_index_path = model_path / "model_index.json"
    if model_index_path.exists():
      config = load_model_index(model_path, model_index_path)
      return config
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

  weight_files = glob.glob(str(model_path/"model*.safetensors"))

  if not weight_files:
    # Try weight for back-compat
    weight_files = glob.glob(str(model_path/"weight*.safetensors"))

  model_class, model_args_class = _get_classes(config=config)

  class ShardedModel(model_class):
    def __init__(self, args):
      super().__init__(args)
      self.shard = Shard(args.shard.model_id, args.shard.start_layer, args.shard.end_layer, args.shard.n_layers)

    def __call__(self, x, *args, **kwargs):
      y = super().__call__(x, *args, **kwargs)
      return y

  model_args = model_args_class.from_dict(config)
  model = ShardedModel(model_args)

  if config.get("model_index", False):
    model.load()
    return model

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

  

  if hasattr(model, "sanitize"):
    weights = model.sanitize(weights)
  if DEBUG >= 8:
    print(f"\n|| {config=} ||\n")

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

async def load_shard(
  model_path: str,
  shard: Shard,
  tokenizer_config={},
  model_config={},
  adapter_path: Optional[str] = None,
  lazy: bool = False,
) -> Tuple[nn.Module, TokenizerWrapper]:
  model = load_model_shard(model_path, shard, lazy, model_config)

  # TODO: figure out a generic solution
  if model.model_type == "llava":
    processor = AutoProcessor.from_pretrained(model_path)
    processor.eos_token_id = processor.tokenizer.eos_token_id
    processor.encode = processor.tokenizer.encode
    return model, processor
  elif hasattr(model, "tokenizer"):
    tokenizer = model.tokenizer
    return model, tokenizer
  else:
    tokenizer = await resolve_tokenizer(model_path)
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

# loading a combined config for all models in the index
def load_model_index(model_path: Path, model_index_path: Path):
  models_config = {}
  with open(model_index_path, "r") as f:
      model_index = json.load(f)
  models_config["model_index"] = True
  models_config["model_type"] = model_index["_class_name"]
  models_config["models"] = {}
  for model in model_index.keys():
    model_config_path = glob.glob(str(model_path / model / "*config.json"))
    if len(model_config_path)>0:
      with open(model_config_path[0], "r") as f:
        model_config = { }
        model_config["model_type"] = model
        model_config["config"] = json.load(f)
        model_config["path"] = model_path / model
        if model_config["path"]/"*model.safetensors":
          model_config["config"].update({"weight_files": list(glob.glob(str(model_config["path"]/"*model.safetensors")))})
        model_config["path"] = str(model_path / model)
        m = {}
        m[model] = model_config
        models_config.update(m)
  return models_config
