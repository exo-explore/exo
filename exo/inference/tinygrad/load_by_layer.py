import argparse
from fnmatch import fnmatch
import os
from pathlib import Path

import numpy as np
from exo.download.hf.hf_helpers import get_allow_patterns
from exo.inference.shard import Shard
import asyncio
import pathlib
import json
from tinygrad.tensor import Tensor
from tinygrad.nn.state import safe_dtypes, load_state_dict

from typing import Any, Union
from tinygrad import Device, Context, nn


from exo.inference.tinygrad.models.llama import Transformer, TransformerShard, convert_from_huggingface3, fix_bf16, sample_logits
from exo.inference.tinygrad.stateful_model import make_prompt_state
from exo.inference.tinygrad.tinygrad_helpers import concat_weights, load
from exo.inference.tokenizers import resolve_tokenizer


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
    print(f"File {fn} not found")
    raise ValueError()
    # assert "model.safetensors" not in str(t), "Model path should not contain model.safetensors"


def load_index_safetensors(path: str) -> dict[str, Any]:
    if os.path.exists(path):
        with open(path, 'r') as f:
            index_data = json.load(f)
        return index_data['weight_map']
    else:
        return None


f = None
data_start = None
metadata = None 
loaded_all = None
layer_dict = None
multi_file=False

def safe_load_by_layer(model_path: str, layer_index: int = -1, l="model.layers.{layer_index}."):
    global f, data_start, metadata, loaded_all, multi_file
    if layer_index >= 0:
        l = l.format(layer_index=layer_index)

    layer_weights = {}
    
    if f is None:
        f, data_start, metadata = safe_load_metadata_single(model_path if not multi_file else os.path.join(model_path, layer_dict[layers_to_load[0]]))
    
    assert f is not None and data_start is not None and metadata is not None, "Failed to load metadata"
    layers_in_file = [ x for x in metadata.keys() if l in x ]

    def load(layer):
        layer_data = metadata[layer]
        f.seek(data_start + (layer_data['data_offsets'][0]))
        size = layer_data['data_offsets'][1] - \
            layer_data['data_offsets'][0]
        data = f.read(size)
        t = Tensor(data, dtype=safe_dtypes[layer_data['dtype']], device="clang").reshape(layer_data['shape'])
        layer_weights[layer] = t

    for layer in layers_in_file:
        print(layer)
        load(layer)
    
    if multi_file:
        layers_to_load = [ x for x in layer_dict.keys() if l in x ]
        while(set(layers_to_load) != set(layers_in_file)):
            layers_to_load = [item for item in layers_to_load if item not in layers_in_file]
            f, data_start, metadata = safe_load_metadata_single(os.path.join(model_path,layer_dict[layers_to_load[-1]]))
            layers_in_file = [ x for x in metadata.keys() if l in x ]
            for layer in layers_to_load:
                print(layer)
                load(layer)
            
    return layer_weights

def build_transformer_2(model_path: str, shard: Shard = None, model_size="8B", verbose=False):
    global layer_dict, multi_file
    linear = nn.Linear
    model = Transformer(**MODEL_PARAMS[model_size]["args"],
                        linear=linear, max_context=8192, jit=True, shard=shard)
    
    fn = os.path.join(model_path, 'model.safetensors.index.json')
    if (os.path.exists(fn)):
        layer_dict = (load_index_safetensors(fn))
        multi_file = True

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

        if shard.end_layer == MODEL_PARAMS[model_size]['args']['n_layers'] - 1:
            weights = safe_load_by_layer(model_path, l="model.norm")
            weights = convert_from_huggingface3(
                weights, model, MODEL_PARAMS[model_size]['args']['n_heads'], MODEL_PARAMS[model_size]['args']['n_kv_heads'])
            weights = fix_bf16(weights)

            load_state_dict(model, weights, strict=False,
                        consume=True, verbose=verbose) 
            
            weights = safe_load_by_layer(model_path, l="output.weight")
            weights = convert_from_huggingface3(
                weights, model, MODEL_PARAMS[model_size]['args']['n_heads'], MODEL_PARAMS[model_size]['args']['n_kv_heads'])
            weights = fix_bf16(weights)

            load_state_dict(model, weights, strict=False,
                        consume=True, verbose=verbose)
        model = TransformerShard(shard, model)
    return model

MODEL_PARAMS = {
  "1B": {
    "args": {
      "dim": 2048, "n_heads": 32, "n_kv_heads": 8, "n_layers": 16, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256, "hidden_dim": 8192,
      "rope_scaling": {"factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0, "original_max_position_embeddings": 8192, "rope_type": "llama3"}, "tie_word_embeddings": True
    }, "files": 1
  }, "3B": {
    "args": {
      "dim": 3072, "n_heads": 24, "n_kv_heads": 8, "n_layers": 28, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256, "hidden_dim": 8192,
      "rope_scaling": {"factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0, "original_max_position_embeddings": 8192, "rope_type": "llama3"}, "tie_word_embeddings": True
    }, "files": 1
  }, "8B": {"args": {"dim": 4096, "n_heads": 32, "n_kv_heads": 8, "n_layers": 32, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256, "hidden_dim": 14336}, "files": 1},
  "70B": {
      "args": {""
        "dim": 8192, "n_heads": 64, "n_kv_heads": 8, "n_layers": 80, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256, "hidden_dim": 28672
      }, 
      "files": 8
    },
  "32B": {
    "args": {
          "dim": 5120, "n_heads": 40, "n_kv_heads": 8, "n_layers": 64, "norm_eps": 1e-5, "rope_theta": 1000000, "vocab_size": 152064, "hidden_dim": 27648
    },
    "files": 17,
    "tokenizer": "Qwen/QwQ-32B-Preview",
  }
}
def build_transformer(model_path: Path, shard: Shard, model_size="8B", device=None):
  # build model
  linear = nn.Linear
  model = Transformer(**MODEL_PARAMS[model_size]["args"], linear=linear, max_context=8192, jit=True, shard=shard)

  # load weights
  if model_path.is_dir():
    if (model_path/"model.safetensors.index.json").exists(): weights = load(str(model_path/"model.safetensors.index.json"), shard)
    elif (model_path/"model.safetensors").exists(): weights = load(str(model_path/"model.safetensors"), shard)
    else: weights = concat_weights([load(str(model_path/f"consolidated.{i:02d}.pth"), shard) for i in range(MODEL_PARAMS[model_size]["files"])], device[0] if isinstance(device, tuple) else device)
  else:
    weights = load(str(model_path), shard)
  weights = convert_from_huggingface3(weights, model, MODEL_PARAMS[model_size]["args"]["n_heads"], MODEL_PARAMS[model_size]["args"]["n_kv_heads"])
  weights = fix_bf16(weights)

  with Context(BEAM=0):
    # replace weights in model
    load_state_dict(model, weights, strict=False, consume=False)  # consume=True
    model = TransformerShard(shard, model)

  return model

async def main(args):
    tokenizer_path = str((Path(args.model_path)))
    tokenizer = await resolve_tokenizer(tokenizer_path)

    model = build_transformer_2(Path(args.model_path), shard=Shard(
        '1', 0, 15, 16), model_size=args.model)
    print(tokenizer.__class__.__name__)
    prompt = "Hello"    
        # Tokenize input prompt
    tokens = np.array(tokenizer.encode(prompt))
    print(f"Input tokens: {tokens}")
    
    # Prepare input for model
    x = tokens.reshape(1, -1)
    x = Tensor(x, dtype=safe_dtypes['F16'])
    
    # Initial embedding and state setup
    h = model.embed(x)
    state = make_prompt_state(x, model)
    m_state = {"start_pos": state.start, "cache": state.cache}
    
    # Process the prompt
    out = model.forward(h, **m_state)
    
    # Extract the last token's logits (prediction scores)
    logits = out.numpy()
    next_token_logits = logits[:, -1, :]
    
    # Track generated tokens
    generated_tokens = []
    
    # Generation loop
    for _ in range(50):
        # Sample next token using temperature and top_p
        next_token = sample_logits(Tensor(next_token_logits).flatten(),0.8, 0, 0.8, 0.0, 0.0).numpy().astype(int)
        
        # Stop if end token reached
        if next_token == tokenizer.eos_token_id:
            break
            
        # Add to generated tokens
        generated_tokens.append(next_token[0])
        print(next_token[0])        
        # Prepare for next iteration
        x_next = Tensor([next_token], dtype=safe_dtypes['F16'])
        h_next = model.embed(x_next)
        
        # Update state position for autoregressive generation
        m_state["start_pos"] += 1
        
        # Get next token prediction
        out = model.forward(h_next, **m_state)
        logits = out.numpy()
        next_token_logits = logits[:, -1, :]
    
    # Decode the generated tokens to text
    # print(generated_tokens)
    response = tokenizer.decode(generated_tokens)
    print(response)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Playground for tinygrad inference.")
    parser.add_argument("--model", type=str, default="1B", help="Model size")
    parser.add_argument("--model-path", type=str,
                        default="models/1B", help="Path to model")
    parser.add_argument("--shard", type=str, default="0", help="Shard index")
    args = parser.parse_args()
    
    model = build_transformer(Path(args.model_path), shard=Shard(
        args.shard, 0, 15, 16), model_size=args.model)
    model2 = build_transformer_2(Path(args.model_path), shard=Shard(
        args.shard, 0, 15, 16), model_size=args.model)

    w_model = nn.state.get_state_dict(model)['norm.weight'][:3].numpy()
    w_model2 = nn.state.get_state_dict(model2)['norm.weight'][:3].numpy()
    assert (w_model == w_model2).all(), "Weights are not equal"
    
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(args))
