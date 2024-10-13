"""
Testing of loading model by layer
"""
import asyncio
import re
import json
import os
from pathlib import Path
from typing import Optional

from exo.download.hf.hf_helpers import (
  get_weight_map,
  download_repo_files
)
from exo.download.hf.hf_shard_download import HFShardDownloader
from exo.inference.shard import Shard

from transformers import AutoModelForCausalLM

async def load_model(
  repo_id: str,
  shard: Shard,
  model_path: Path,
  weight_map: Optional[dict]
) -> Optional[AutoModelForCausalLM]:
  """
  load model by layer and safetensors
  return causal llm automodel with only requested layers, if weight maps
  if no weight map, return and load the whole model
  """
  print("load_model called")
  model_st_snapshot = model_path/"model.safetensors.index.json"

  if weight_map:
    layer_weight_map = {}
    skip_layers = []

    for wname, wtensor in weight_map.items():
      # get layer number
      layer_rgx = r'^model\.layers\.(\d+)\.*'
      layer_found = re.findall(layer_rgx, wname)
      print(f"wname: {wname}")
      if layer_found:
        print(f"layer_found: {layer_found}")
        # slice up layer map to start and end layers
        # from shard
        layer_idx = int(layer_found[0])
        if shard.start_layer <= layer_idx <= shard.end_layer:
          layer_weight_map[wname] = wtensor
        else:
          skip_layers.append(wname)
          print(f"SKIPPING LAYER {layer_idx}")

      if wname not in skip_layers:
        print(f"adding non-layer: {wname}")
        layer_weight_map[wname] = wtensor

    # will manipulate current model.safetensors.index.json
    # but set back at end of inference
    print(layer_weight_map)

    # rewrite model.safetensors.index.json
    try:
      # call download repo files again to reload original safetensors json
      #os.remove(model_st_snapshot)

      #await download_repo_files(
      #  repo_id=shard.model_id,
      #  revision="main",
      #  allow_patterns="model.safetensors.index.json")

      mst_json = {}
      with open(model_st_snapshot, "r") as mst_file:
        mst_json = json.load(mst_file)

        mst_json["weight_map"] = layer_weight_map

        print(f"mst_json: {json.dumps(mst_json, indent=4)}")

        os.remove(model_st_snapshot)

        with open(model_st_snapshot, "w") as mst_file:
          json.dump(mst_json, mst_file, indent=4)
          print(f"{model_st_snapshot} rewritten with {shard.n_layers} weights")
    except Exception as err:
      print(f"err: {err}")
      raise

  else:
    print("weight_map not found, loading whole model")

  # load model with layer edits
  # or whole model if no weight_map
  shard_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    offload_buffers=True
  )

  # have to clear out edited model safetensors mst_json
  os.remove(model_st_snapshot)

  return shard_model


async def test_split_model(
  model_id: str,
  start_layer: int,
  end_layer: int,
  n_layers: int
):
  """
  Test to load split models
  """

  shard = Shard(
    model_id=model_id,
    start_layer=start_layer,
    end_layer=end_layer-1,
    n_layers=n_layers
  )

  # remove old weight json if present


  print(f"loading shard: {shard}")
  shard_downloader = HFShardDownloader()
  model_path = await shard_downloader.ensure_shard(shard)
  weight_map = await get_weight_map(model_id)

  await load_model(
    model_id,
    shard,
    model_path,
    weight_map
  )

if __name__ == "__main__":
  #Qwen/Qwen2.5-3B
  try:
    print("\n-------- Test Qwen/Qwen2.5-3B-Instruct ----------\n")
    asyncio.run(test_split_model(
      "Qwen/Qwen2.5-3B-Instruct",
      0,
      18,
      36
    ))
  except Exception as err:
    print(f"\n\n !!!!!!!!!!! meta-llama/Llama-3.2-1B-Instruct TEST FAILED \n{err}\n")

  # unsloth/Meta-Llama-3.1-8B-Instruct
    #try:
    #  print("\n-------- Test unsloth/Meta-Llama-3.1-8B-Instruct ----------\n")
    #  asyncio.run(test_split_model(
    #    "unsloth/Meta-Llama-3.1-8B-Instruct",
    #    0,
    #    1,
    #    32
    #  ))
    #except Exception as err:
    #  print(f"\n\n !!!!!!!!!!! meta-llama/Llama-3.2-1B-Instruct TEST FAILED \n{err}\n")
