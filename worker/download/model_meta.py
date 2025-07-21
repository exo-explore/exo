import json
from typing import Annotated, Dict, Optional

import aiofiles
from huggingface_hub import model_info
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError
from pydantic import BaseModel, Field

from shared.types.models import ModelMetadata
from worker.download.download_utils import (
  ModelSafetensorsIndex,
  download_file_with_retry,
  ensure_exo_tmp,
)
from worker.download.model_cards import MODEL_CARDS


class ConfigData(BaseModel):
  num_hidden_layers: Optional[Annotated[int, Field(ge=0)]]
  num_layers: Optional[Annotated[int, Field(ge=0)]]
  n_layer: Optional[Annotated[int, Field(ge=0)]]

async def get_config_data(model_id: str) -> Optional[ConfigData]:
  """Downloads and parses config.json for a model."""
  try:
    model_card = MODEL_CARDS[model_id]
    target_dir = (await ensure_exo_tmp())/model_card.repo_id.replace("/", "--")
    config_path = await download_file_with_retry(model_card.repo_id, "main", "config.json", target_dir, lambda curr_bytes, total_bytes: print(f"Downloading config.json for {model_id}: {curr_bytes}/{total_bytes}"))
    async with aiofiles.open(config_path, 'r') as f:
      return ConfigData.model_validate_json(await f.read())
  except EntryNotFoundError:
    print(f"Warning: config.json not found for {model_id}. Layers/type from config unavailable.")
  except json.JSONDecodeError:
    print(f"Error: Failed to parse config.json for {model_id}.")
  except Exception as e:
    print(f"Error: Error processing config.json for {model_id}: {e}")
  return None

def get_num_layers(config_data: Optional[ConfigData], model_id: str) -> Optional[int]:
  """Extracts number of layers from config data."""
  if not config_data:
    return None

  if config_data.num_hidden_layers is not None:
    return config_data.num_hidden_layers
  if config_data.num_layers is not None:
    return config_data.num_layers
  if config_data.n_layer is not None:
    return config_data.n_layer

  print(f"Warning: No known layer key or valid number in config.json for {model_id}. Config: {config_data.model_dump_json()}")
  return None

async def get_safetensors_size(model_id: str) -> Optional[int]:
  """Gets model size from safetensors index or falls back to HF API."""
  try:
    model_card = MODEL_CARDS[model_id]
    target_dir = (await ensure_exo_tmp())/model_card.repo_id.replace("/", "--")
    index_path = await download_file_with_retry(model_card.repo_id, "main", "model.safetensors.index.json", target_dir, lambda curr_bytes, total_bytes: print(f"Downloading model.safetensors.index.json for {model_id}: {curr_bytes}/{total_bytes}"))
    async with aiofiles.open(index_path, 'r') as f:
      index_data = ModelSafetensorsIndex.model_validate_json(await f.read())

    metadata = index_data.metadata
    if metadata is not None:
      return metadata.total_size
    print(f"Warning: Could not extract total_size from safetensors index metadata for {model_id}. Metadata: {index_data.model_dump_json()}")

  except EntryNotFoundError:
    print(f"Warning: model.safetensors.index.json not found for {model_id}.")
  except json.JSONDecodeError:
    print(f"Error: Failed to parse model.safetensors.index.json for {model_id}.")
  except Exception as e:
    print(f"Error: Error processing model.safetensors.index.json for {model_id}: {e}")

  print(f"Warning: Could not determine safetensors total size from index for {model_id}. Falling back to model_info API call.")
  try:
    info = model_info(model_id)
    if info.safetensors is not None:
        return info.safetensors.total
    print(f"Warning: Could not get safetensors total size from model_info API for {model_id}. Safetensors info: {info}")
  except HfHubHTTPError as e:
    print(f"Error: HTTP Error while fetching model info from API for {model_id}: {e}")
  except Exception as e:
    print(f"Error: Error getting total size from huggingface info API for {model_id}: {e}")
  return None

_model_meta_cache: Dict[str, ModelMetadata] = {}
async def get_model_meta(model_id: str) -> ModelMetadata:
  if model_id in _model_meta_cache:
    return _model_meta_cache[model_id]
  model_meta = await _get_model_meta(model_id)
  _model_meta_cache[model_id] = model_meta
  return model_meta

async def _get_model_meta(model_id: str) -> ModelMetadata:
  """Fetches storage size and number of layers for a Hugging Face model, returns Pydantic ModelMeta."""
  model_card = MODEL_CARDS[model_id]
  num_layers_val: Optional[int] = None
  mem_size_bytes_val: Optional[int] = None
  try:
    config_data = await get_config_data(model_id)
    # get_num_layers is synchronous
    num_layers_val = get_num_layers(config_data, model_id)
    mem_size_bytes_val = await get_safetensors_size(model_id)

  except HfHubHTTPError as e:
    print(f"Error: HTTP Error encountered for '{model_id}': {e}")
  except Exception as e:
    print(f"Error: Unexpected error during metadata fetching for '{model_id}': {e}")
  
  # Fallbacks for missing metadata
  if mem_size_bytes_val is None:
    print(f"Warning: Could not determine model size for {model_id}. Defaulting to 0 bytes.")
    mem_size_bytes_val = 0
  if num_layers_val is None:
    print(f"Warning: Could not determine number of layers for {model_id}. Defaulting to 0 layers.")
    num_layers_val = 0

  return ModelMetadata(
    model_id=model_id,
    pretty_name=model_card.name,
    storage_size_kilobytes=mem_size_bytes_val // 1024,
    n_layers=num_layers_val,
  )
