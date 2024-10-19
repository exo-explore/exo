"""
HuggingFace Safetensor Shard
Sharding of safetensors to only use weights of models needed
"""
import os
import shutil
from safetensors import safe_open
from safetensors.torch import save_file

class HFSafeTensorShard:
  def __init__(self, model_folder, start_layer, end_layer):
    self.model_folder = model_folder
    self.start_layer = start_layer
    self.end_layer = end_layer
    self.safetensor_path = self.get_safetensor_path()
    self.backup_path = self.safetensor_path + ".backup"

  def get_safetensor_path(self):
    try:
      for file_name in os.listdir(self.model_folder):
        if file_name.endswith(".safetensors"):
          return os.path.join(self.model_folder, file_name)
      raise FileNotFoundError("No safetensors file found in the provided model folder.")
    except Exception as err:
      print(f"Error in get_safetensor_path: {err}")
      raise

  def backup_safetensor(self):
    try:
      if not os.path.exists(self.backup_path):
        shutil.copy(self.safetensor_path, self.backup_path)
        print(f"Backup created at {self.backup_path}")
      else:
        print("Backup already exists. Skipping backup.")
    except Exception as err:
      print(f"Error in backup_safetensor: {err}")
      raise

  def modify_safetensor(self):
    # Ensure the safetensor is backed up before modifying
    self.backup_safetensor()

    try:
      with safe_open(self.safetensor_path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        new_tensors = {}

        # Iterate over tensors, including only those within the specified layer range
        for key in f.keys():
          layer_number = self.extract_layer_number(key)
          if self.start_layer <= layer_number <= self.end_layer:
            new_tensors[key] = f.get_tensor(key)
          else:
            print(f"Excluding layer {layer_number}: {key}")

        # Save the modified safetensor
        save_file(new_tensors, self.safetensor_path, metadata)
        print(f"Safetensor modified and saved to {self.safetensor_path}")
    except Exception as err:
      print(f"Error modifying safetensor: {err}")
      raise

  def extract_layer_number(self, key):
    """
    Extract the layer number from a tensor key.
    This function assumes keys follow the format 'transformer.h.<layer>.<subkey>'.
    """
    try:
      parts = key.split(".")
      layer_idx = next(i for i, part in enumerate(parts) if part.startswith("h"))
      return int(parts[layer_idx + 1])
    except (IndexError, ValueError) as err:
      print(f"Error extracting layer number from key '{key}': {err}")
      return -1

  def restore_backup(self):
    """
    Restore the original safetensor from the backup file.
    This is useful when you want to reset to the original before making new modifications.
    """
    try:
      if os.path.exists(self.backup_path):
        shutil.copy(self.backup_path, self.safetensor_path)
        print(f"Safetensor restored from backup at {self.backup_path}")
      else:
        print("No backup found. Cannot restore.")
    except Exception as err:
      print(f"Error in restore_backup: {err}")
      raise
