from tinygrad.nn.state import safe_load, torch_load
from tinygrad import Tensor
from pathlib import Path
import json
from typing import List
from exo.inference.shard import Shard
from exo.helpers import DEBUG
from exo.download.hf.hf_helpers import get_allow_patterns
from fnmatch import fnmatch
import re


# **** helper functions ****
def concat_weights(models, device=None):
  def convert(name) -> Tensor:
    disk_tensors: List[Tensor] = [model[name] for model in models]
    if len(disk_tensors) == 1 or len(disk_tensors[0].shape) == 1:
      return disk_tensors[0].to(device=device)
    axis = 1 if name.endswith(".attention.wo.weight") or name.endswith(".feed_forward.w2.weight") else 0
    lazy_tensors = [data.to(device=device) for data in disk_tensors]
    return lazy_tensors[0].cat(*lazy_tensors[1:], dim=axis)

  return {name: convert(name) for name in {name: None for model in models for name in model}}


def load(fn: str, shard: Shard):
  if fn.endswith('.index.json'):
    with open(fn) as fp:
      weight_map = json.load(fp)['weight_map']
    parts = {}
    filtered_weight_map = {}
    allow_patterns = get_allow_patterns(weight_map, shard)
    for k, n in weight_map.items():
      if allow_patterns is not None and not any(fnmatch(n, r) for r in allow_patterns):
        continue
      if k.startswith("model.layers."):
        layer_num = int(k.split('.')[2])
        if layer_num < shard.start_layer or layer_num > shard.end_layer:
          continue

      parts[n] = load(str(Path(fn).parent/Path(n).name), shard)
      filtered_weight_map[k] = n
    if DEBUG >= 2: print(f"Excluded model param keys for {shard=}: {sorted(set(weight_map.keys()) - set(filtered_weight_map.keys()))}")
    return {k: parts[n][k] for k, n in filtered_weight_map.items()}
  elif fn.endswith(".safetensors"):
    weight_map = safe_load(fn)
    for k in list(weight_map):
      if (n := re.search(r"\.(\d+)\.", k)) and not (shard.start_layer <= int(n.group(1)) <= shard.end_layer):
          del weight_map[k]
    return weight_map
  else:
    return torch_load(fn)
