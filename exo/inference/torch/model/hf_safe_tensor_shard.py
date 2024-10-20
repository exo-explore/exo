"""
HuggingFace Safetensor Shard
Sharding of safetensors to only use weights of models needed
"""
import os
import shutil
import json

from typing import Optional
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

import torch

from exo.inference.shard import Shard
from exo.helpers import DEBUG
from exo.inference.torch.utils import extract_layers

class HFSafeTensorShard:
  def __init__(self, model_path: Path, shard: Shard):
    self.model_path = model_path
    self.shard = shard
    self.safetensors_path = self.get_safetensors()
    self.safetensor_index_path = f"{self.model_path}/model.safetensors.index.json"
    self.metadata = {
      "metadata": {
        "total_size": 0
      },
      "weight_map": {}
    }

  def get_safetensors(self) -> list:
    safetensors_path = []
    try:
      for file_name in os.listdir(self.model_path):
        if file_name.endswith(".safetensors"):
          safetensor_path = os.path.join(self.model_path, file_name)
          if safetensor_path not in safetensors_path:
            safetensors_path.append(safetensor_path)
    except Exception as err:
      print(f"Error in get_safetensor_path: {err}")
      raise

    return safetensors_path

  def backup_safetensor(self):
    try:
      for safetensor_path in self.safetensors_path:
        backup_path = safetensor_path+".backup"
        if not os.path.exists(backup_path):
          shutil.copy(safetensor_path, backup_path)

          if DEBUG >= 4:
            print(f"Backup created at {backup_path}")
    except Exception as err:
      print(f"Error in backup_safetensor: {err}")
      raise

  def modify_safetensor(self):
    """
    Extract needed weights for layers from safetensor files
    and create a new safetensor with same names
    """
    try:
      self.backup_safetensor()
      safetensor_is_used = False
      for safetensor_path in self.safetensors_path:
        initial_size = os.path.getsize(safetensor_path)
        with safe_open(safetensor_path, framework="pt") as f:
          metadata = f.metadata()
          new_tensors = {}

          # Iterate over tensors, including only those within the specified layer range
          for key in f.keys():
            layer_number = self.extract_layer_number(key)
            if self.shard.start_layer <= layer_number <= self.shard.end_layer:
              if DEBUG >= 4:
                print(f"modify_safetensor [{layer_number}] extracting {key}")
              new_tensors[key] = f.get_tensor(key)
              safetensor_is_used = True

          # Save the modified safetensor
          if safetensor_is_used:
            save_file(new_tensors, safetensor_path, metadata)
            modified_size = os.path.getsize(safetensor_path)

            if DEBUG >= 4:
              print(f"Safetensor modified and saved to {safetensor_path}")
              print(f"Initial size: {initial_size / (1024**3):.2f} GB")
              print(f"Modified size: {modified_size / (1024**3):.2f} GB")
          else:
            # remove unused safetensors
            os.remove(safetensor_path)

            if DEBUG >= 4:
              print(f"Removed safetensor: {safetensor_path}")
    except Exception as err:
      print(f"Error modifying safetensor: {err}")
      raise

  def extract_layer_number(self, key):
    """
    Extract the layer number from a tensor key.
    This function assumes keys follow the format 'model.layers.<idx>.<type>'.
    """
    try:
      parts = key.split(".")
      layer_idx = 0
      if parts[0] == "model" and parts[1] == "layers":
        layer_idx = int(parts[2])
      return layer_idx
      #layer_idx = next(i for i, part in enumerate(parts) if part.startswith("h"))
      #return int(parts[layer_idx + 1])
    except (IndexError, ValueError) as err:
      print(f"Error extracting layer number from key '{key}': {err}")
      return -1

  def create_safetensor_index(self):
    """
    Creates a model.safetensors.index.json file from a list of safetensor files.

    Args:

    Raises:
    """
    if os.path.exists(self.safetensor_index_path):
      backup_index_path = f"{self.model_path}/model.safetensors.index.json.backup"
      if not os.path.exists(backup_index_path):
        shutil.copy(self.safetensor_index_path, backup_index_path)

        if DEBUG >= 4:
          print(f"backed up index json {self.safetensor_index_path}")

    if self.safetensors_path:
      # initialize the metadata and weight_map 
      for safetensor_file in self.safetensors_path:
        # use the safetensor file name as the shard_name
        shard_name = os.path.basename(safetensor_file)

        # open the safetensor file to read the metadata
        with safe_open(safetensor_file, framework="pt", device="cpu") as f:
          # get tensor names
          tensor_names = f.keys()

          # collect metadata for each tensor
          for name in tensor_names:
            tensor_data = f.get_tensor(name)
            shape = tensor_data.shape
            dtype = tensor_data.dtype

            # calculate the tensor size in bytes based on dtype
            total_elements = 1
            for dim in shape:
              total_elements *= dim

            if dtype == torch.float32:
              element_size = 4
            elif dtype == torch.float16 or dtype == torch.bfloat16:
              element_size = 2
            # extend this to support more data types if needed
            else:
              raise ValueError(f"unsupported dtype: {dtype}")

            tensor_size = total_elements * element_size
            self.metadata["metadata"]["total_size"] += tensor_size

            # add to weight_map, mapping the tensor to the shard (file) name
            self.metadata["weight_map"][name] = shard_name

      # write the metadata and weight map to the index file
      with open(self.safetensor_index_path, "w") as f:
        json.dump(self.metadata, f, indent=4)

      if DEBUG >= 4:
        print(f"created new {self.safetensor_index_path}")
    else:
      print("No safetensor files provided.")

  def shard_safetensor_index(self, weight_map: Optional[dict] = None):
    if weight_map is None:
      weight_map = self.metadata["weight_map"]

    layer_weight_map = extract_layers(
      weight_map,
      self.shard
    )

    # rewrite model.safetensors.index.json for only needed layers
    try:
      mst_json = {}
      with open(self.safetensor_index_path, "r") as mst_file:
        mst_json = json.load(mst_file)
        mst_json["weight_map"] = layer_weight_map

      if DEBUG >= 4:
        print(f"new safetensor index\n{json.dumps(mst_json, indent=4)}\n")

      os.remove(self.safetensor_index_path)

      with open(self.safetensor_index_path, "w") as mst_file:
        json.dump(mst_json, mst_file, indent=4)
    except Exception as err:
      print(f"err: {err}")
      raise

  def restore_backups(self):
    """
    Restore the original safetensor and index json, if any, from the backup file.
    """
    try:
      for safetensor_path in self.safetensors_path:
        backup_path = safetensor_path+".backup"
        if os.path.exists(backup_path):
          os.remove(safetensor_path)
          shutil.copy(backup_path, safetensor_path)
          os.remove(backup_path)

          if DEBUG >= 4:
            print(f"Safetensor restored from backup at {backup_path}")

      backup_index_path = self.safetensor_index_path+".backup"
      if os.path.exists(backup_index_path):
        os.remove(self.safetensor_index_path)
        shutil.copy(backup_index_path, self.safetensor_index_path)
        os.remove(backup_index_path)

        if DEBUG >= 4:
          print(f"Safetensor index JSON restored from backup at {backup_index_path}")
    except Exception as err:
      print(f"Error in restore_backup: {err}")
      raise

