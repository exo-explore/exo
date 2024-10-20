"""
Testing of loading model by layer
"""
import asyncio
import re
import json
import os
from pathlib import Path
from typing import Optional

import torch

from transformers.modeling_utils import offload_weight

from exo.download.hf.hf_helpers import get_weight_map
from exo.download.hf.hf_shard_download import HFShardDownloader
from exo.inference.shard import Shard
from exo.inference.torch.utils import print_cuda_vram_stats

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

def load_model(
  repo_id: str,
  shard: Shard,
  model_path: Path,
  weight_map: Optional[dict],
  device: Optional[str] = "cuda"
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
    non_layer_weights = []

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
        non_layer_weights.append((wname, wtensor))

    non_layer_weights = sorted(non_layer_weights, key=lambda x: x[1])

    print(f"sorted non_layer_weights: {non_layer_weights}")

    if shard.is_first_layer():
      # this assumes at max only one first weight non-layer for model
      first_weight = non_layer_weights[0]
      layer_weight_map[first_weight[0]] = first_weight[1]
    elif shard.is_last_layer():
      last_weights = non_layer_weights[1:]
      for last_weight in last_weights:
        layer_weight_map[last_weight[0]] = last_weight[1]

    # rewrite model.safetensors.index.json
    try:
      mst_json = {}
      with open(model_st_snapshot, "r") as mst_file:
        mst_json = json.load(mst_file)
        mst_json["weight_map"] = layer_weight_map

        print(f"mst_json: {json.dumps(mst_json, indent=4)}")

        os.remove(model_st_snapshot)

        with open(model_st_snapshot, "w") as mst_file:
          json.dump(mst_json, mst_file, indent=4)
    except Exception as err:
      print(f"err: {err}")
      raise

  else:
    print("weight_map not found, loading whole model")

  # setup the weight range for init_weights
  shard_num_hidden_layers = shard.end_layer - shard.start_layer
  print(f"Setting up LLM config with {shard_num_hidden_layers} hidden layers")
  llm_config = AutoConfig.from_pretrained(
    pretrained_model_name_or_path=model_path,
    device_map="cuda",
    offload_buffers=True,
    local_files_only=True,
    num_hidden_layers=shard_num_hidden_layers
  )

  # load model with layer edits
  # or whole model if no weight_map
  print(f"Loading sharded AutoModelForCausalLM from {model_path}")
  shard_model = AutoModelForCausalLM.from_config(llm_config).to(device)

  print("Loading tokenizer")
  tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_path,
    local_files_only=True,
  )

  if torch.cuda.is_available() and device == "cuda":
    print_cuda_vram_stats()

  prompt = "In a single word only, what color is a red apple?"

  model_inputs = tokenizer(
    [prompt],
    return_tensors="pt"
  )

  generated_ids = shard_model.generate(
    model_inputs.input_ids.to(device),
    attention_mask=model_inputs.attention_mask.to(device),
    max_new_tokens=512,
    do_sample=True
  )

  generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(
      model_inputs.input_ids,
      generated_ids
    )
  ]

  response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

  print(f"Prompt: {prompt}\n")
  print(f"Response: {response}\n")

  print_ram_stats()

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

  print(f"loading shard: {shard}")
  shard_downloader = HFShardDownloader()
  model_path = await shard_downloader.ensure_shard(shard)
  weight_map = await get_weight_map(model_id)

  load_model(
    model_id,
    shard,
    model_path,
    weight_map
  )

if __name__ == "__main__":
  #Qwen/Qwen2.5-3B
  #try:
  #  print("\n-------- Test Qwen/Qwen2.5-3B-Instruct ----------\n")
  #  asyncio.run(test_split_model(
  #    "Qwen/Qwen2.5-3B-Instruct",
  #    0,
  #    6,
  #    36
  #  ))
  #except Exception as err:
  #  print(f"\n\n !!!!!!!!!!! Qwen/Qwen2.5-3B-Instruct TEST FAILED \n{err}\n")

  # unsloth/Meta-Llama-3.1-8B-Instruct
  try:
    print("\n-------- Test unsloth/Meta-Llama-3.1-8B-Instruct ----------\n")
    asyncio.run(test_split_model(
      "unsloth/Meta-Llama-3.1-8B-Instruct",
      0,
      6,
      32
    ))
  except Exception as err:
    print(f"\n\n !!!!!!!!!!! meta-llama/Llama-3.2-1B-Instruct TEST FAILED \n{err}\n")
