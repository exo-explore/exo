"""
Testing of loading model by layer
"""
import asyncio
import re

from exo.download.hf.hf_helpers import get_weight_map
from exo.download.hf.hf_shard_download import HFShardDownloader
from exo.inference.shard import Shard

from typing import Optional, Union, Tuple

from transformers import AutoModel

async def load_model(
  repo_id: str,
  shard: Shard
) -> Optional[AutoModel]:
  """
  load model by layer and safetensors
  """

  shard_downloader = HFShardDownloader()
  model_path = await shard_downloader.ensure_shard(shard)
  weight_map = await get_weight_map(repo_id)

  if weight_map:
    for wname, wtensor in weight_map.items():
      # get layer number
      layer_rgx = r'^model\.layers\.(\d+)\.(\w+)\.(\w+)$'
      layer_found = re.findall(layer_rgx, wname)
      if layer_found:
        try:
          layer_idx = int(layer_found[0][0])
          print(f"layer_idx: {layer_idx}")
          if shard.start_layer <= layer_idx <= shard.end_layer:
            print(f"wtensor: {wtensor}")

            # move to local .tmp folder that can be removed later
            # check if files not already there, if there, reuse
            # create automodel with rest of layers
            # lm_head needed at end
        except Exception as err:
          print(f"err: {err}")

async def test_split_model(model_id: str, n_layers: int):
  """
  Test to load split models
  """

  shard = Shard(
    model_id=model_id,
    start_layer=0,
    end_layer=n_layers-1,
    n_layers=n_layers
  )

  await load_model(
    model_id,
    shard
  )

if __name__ == "__main__":
  try:
    print("\n-------- Test unsloth/Meta-Llama-3.1-8B-Instruct ----------\n")
    asyncio.run(test_split_model(
      "unsloth/Meta-Llama-3.1-8B-Instruct",
      32
    ))
  except Exception as err:
    print(f"\n\n !!!!!!!!!!! meta-llama/Llama-3.2-1B-Instruct TEST FAILED \n{err}\n")
