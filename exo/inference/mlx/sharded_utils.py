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
from transformers import AutoProcessor

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
  for wf in weight_files:
    weights.update(mx.load(wf))

  model_class, model_args_class = _get_classes(config=config)

  model_args = model_args_class.from_dict(config)
  model = model_class(model_args)

  if hasattr(model, "sanitize"):
    weights = model.sanitize(weights)

  if (quantization := config.get("quantization", None)) is not None:
    nn.quantize(
      model,
      **quantization,
      class_predicate=None,
    )

  model.load_weights(list(weights.items()), strict=True)

  if not lazy:
    mx.eval(model.parameters())

  model.eval()
  return model


async def snapshot_download_async(*args, **kwargs):
  func = partial(snapshot_download, *args, **kwargs)
  return await asyncio.get_event_loop().run_in_executor(None, func)


async def get_model_path(path_or_hf_repo: str, revision: Optional[str] = None) -> Path:
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
        await snapshot_download_async(
          repo_id=path_or_hf_repo,
          revision=revision,
          allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.py",
            "tokenizer.model",
            "*.tiktoken",
            "*.txt",
          ],
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
  model_path = await get_model_path(path_or_hf_repo)

  model = load_model_shard(model_path, shard, lazy, model_config)
  if adapter_path is not None:
    model = apply_lora_layers(model, adapter_path)
    model.eval()

  # TODO: figure out a generic solution
  if model.model_type == "llava":
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor
  else:
    tokenizer = load_tokenizer(model_path, tokenizer_config)
    return model, tokenizer
