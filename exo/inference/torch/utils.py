"""
Utility functions to be used by inference engine
and model
"""
import re

from exo.inference.shard import Shard

def extract_layers(
  weight_map: dict,
  shard: Shard
) -> dict:
  """
  Extract layers from weight map in range

  Args:

  Returns:
  """

  layer_rgx = r'^model\.layers\.(\d+)\.*'
  layer_weight_map = {}
  non_layer_weights = []

  for wname, wtensor in weight_map.items():
    layer_found = re.findall(layer_rgx, wname)
    if layer_found:
      layer_idx = int(layer_found[0])
      if shard.start_layer <= layer_idx <= shard.end_layer:
        layer_weight_map[wname] = wtensor
    else:
      non_layer_weights.append((wname, wtensor))

  non_layer_weights = sorted(non_layer_weights, key=lambda x: x[1])

  print(non_layer_weights)
  print(f"first: {shard.is_first_layer()}")
  print(f"last: {shard.is_last_layer()}")

  if shard.is_first_layer():
    # this assumes at max only one first weight non-layer for model
    first_weight = non_layer_weights[0]
    layer_weight_map[first_weight[0]] = first_weight[1]
  elif shard.is_last_layer():
    last_weights = non_layer_weights[1:]
    for last_weight in last_weights:
      layer_weight_map[last_weight[0]] = last_weight[1]

  return layer_weight_map
