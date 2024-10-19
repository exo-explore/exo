"""
Create a model.safetensors.index.json from safetensors
"""
import json
import os

import asyncio

from safetensors import safe_open

from exo.download.hf.hf_shard_download import HFShardDownloader
from exo.inference.shard import Shard

import torch

def create_safetensor_index(safetensor_files: list, index_file: str):
  """
  Creates a model.safetensors.index.json file from a list of safetensor files.

  Args:
      safetensor_files (list): List of paths to the safetensor files.
      index_file (str): Path where the index JSON file should be saved.

  Raises:
      ValueError: If an unsupported data type is encountered.
  """
  if safetensor_files:
    # Initialize the metadata and weight_map
    metadata = {
      "metadata": {
        "total_size": 0
      },
      "weight_map": {}
    }

    for safetensor_file in safetensor_files:
      # Use the safetensor file name as the shard_name
      shard_name = os.path.basename(safetensor_file)

      # Open the safetensor file to read the metadata
      with safe_open(safetensor_file, framework="pt") as f:
        # Get tensor names
        tensor_names = f.keys()

        # Collect metadata for each tensor
        for name in tensor_names:
          tensor_data = f.get_tensor(name)
          print(f"tensor_data: {tensor_data}")
          shape = tensor_data.shape
          dtype = tensor_data.dtype
          print(f"shape: {shape}")
          print(f"dtype: {str(dtype) == "torch.bfloat16"}")

          # Calculate the tensor size in bytes based on dtype
          total_elements = 1
          for dim in shape:
            total_elements *= dim

          if dtype == torch.float32:
            element_size = 4
          elif dtype == torch.float16 or dtype == torch.bfloat16:
            element_size = 2
          # Extend this to support more data types if needed
          else:
            raise ValueError(f"Unsupported dtype: {dtype}")

          tensor_size = total_elements * element_size
          metadata["metadata"]["total_size"] += tensor_size

          # Add to weight_map, mapping the tensor to the shard (file) name
          metadata["weight_map"][name] = shard_name

    # Write the metadata and weight map to the index file
    with open(index_file, "w") as f:
      json.dump(metadata, f, indent=4)

    print(f"Index file created: {index_file}")
  else:
    print("No safetensor files provided.")


async def main():
  """
  Main asynchronous function to download the model shard and create an index file for safetensors.

  This function downloads a model shard from Hugging Face, identifies safetensor files, and
  generates a corresponding index file using the `create_safetensor_index` function.
  """
  start_layer = 3
  end_layer = 5

  # Create a Shard object
  shard = Shard(
    model_id="meta-llama/Llama-3.2-1B-Instruct",
    start_layer=start_layer,
    end_layer=end_layer-1,
    n_layers=32
  )

  print(f"Loading shard: {shard}")
  shard_downloader = HFShardDownloader()

  # Ensure shard is downloaded
  model_path = await shard_downloader.ensure_shard(shard)

  # Collect all safetensor files from the model path
  safetensor_files = [
    os.path.join(model_path, file_name)
    for file_name in os.listdir(model_path) if file_name.endswith(".safetensors")
  ]

  # Create the index file
  if safetensor_files:
    create_safetensor_index(safetensor_files, os.path.join(model_path, "model.safetensors.index.json"))
  else:
    print("No safetensor files found in the model path.")


if __name__ == "__main__":
  asyncio.run(main())
