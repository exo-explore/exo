import argparse
from fnmatch import fnmatch
import os
from pathlib import Path
from exo.download.hf.hf_helpers import get_allow_patterns
from exo.inference.shard import Shard
import asyncio
import pathlib
import json
from tinygrad.tensor import Tensor
from tinygrad.nn.state import safe_dtypes, load_state_dict

from typing import Any, Union
from tinygrad import Device, Context, nn

from exo.inference.tinygrad.models.llama import Transformer, TransformerShard, convert_from_huggingface3, fix_bf16
from exo.inference.tinygrad.tinygrad_helpers import concat_weights, load


def safe_load_metadata_single(fn: Union[str, pathlib.Path]) -> tuple[Tensor, int, dict[str, Any]]:
    if not ".safetensors" in str(fn):
        fn = f"{fn}/model.safetensors"
    
    print(f"Loading {fn}")  
    if (os.path.exists(fn)):
        f = open(fn, "rb")
        size = f.read(8)
        data_start = int.from_bytes(size, "little")
        raw_data = f.read(data_start)
        data_start += 8
        print(f"Loading {fn} with size {data_start}")
        return f, data_start, json.loads(raw_data.decode('utf-8'))
    
    raise ValueError(f"File {fn} not found")
    # assert "model.safetensors" not in str(t), "Model path should not contain model.safetensors"

def safe_load_metadata_multifile(fn: Union[str, pathlib.Path], l) -> tuple[Tensor, int, dict[str, Any], bool]:

    weight_map = json.load(open(fn, "r"))['weight_map']
    layers_files = []
    print(l)
    for k, v in weight_map.items():
        if l in k:
            layers_files.append(v)
    if len(set(layers_files)) > 1:
        return *safe_load_metadata_single( f"{fn[:-28]}/{layers_files[0]}"), False

    return *safe_load_metadata_single( f"{fn[:-28]}/{layers_files[0]}"), True

def safe_load_by_layer(model_path: str, layer_index: int = -1, l="model.layers.{layer_index}."):
    if layer_index >= 0:
        l = l.format(layer_index=layer_index)


    fn = f"{model_path}/model.safetensors.index.json"
    # assert os.path.exists(fn), "safetensors.index.json not exists"

    layer_weights = {}
    f, data_start, metadata, loaded_all = (*safe_load_metadata_single(model_path), True)  if not os.path.exists(fn) else  safe_load_metadata_multifile(fn, l)
    last_layer = ""
    if not loaded_all:
        last_layer = f"models.layers.{layer_index - 1}"
    for k in metadata.keys():
            if l in k or (not loaded_all and last_layer in k):
                layer_data = metadata[k]
                f.seek(data_start + (layer_data['data_offsets'][0]))
                size = layer_data['data_offsets'][1] - \
                    layer_data['data_offsets'][0]
                data = f.read(size)
                t = Tensor(data, dtype=safe_dtypes[layer_data['dtype']]).reshape(layer_data['shape'])
                layer_weights[k] = t
    return layer_weights

def build_transformer_2(model_path: str, shard: Shard = None, model_size="8B", verbose=False):

    linear = nn.Linear
    model = Transformer(**MODEL_PARAMS[model_size]["args"],
                        linear=linear, max_context=8192, jit=True, shard=shard)
    with Context(BEAM=0):
        # load embedings
        if shard.start_layer == 0:
            weights = safe_load_by_layer(model_path, l="model.embed_tokens")
            weights = convert_from_huggingface3(
                weights, model, MODEL_PARAMS[model_size]['args']['n_heads'], MODEL_PARAMS[model_size]['args']['n_kv_heads'])
            weights = fix_bf16(weights)
            load_state_dict(model, weights, strict=False,
                            consume=True, verbose=verbose)

        for i in range(shard.start_layer, shard.end_layer + 1):
            weights = safe_load_by_layer(model_path, i)
            weights = convert_from_huggingface3(
                weights, model,  MODEL_PARAMS[model_size]['args']['n_heads'], MODEL_PARAMS[model_size]['args']['n_kv_heads'])
            weights = fix_bf16(weights)
            load_state_dict(model, weights, strict=False,
                            consume=True, verbose=verbose)

        if shard.end_layer == MODEL_PARAMS[model_size]['args']['