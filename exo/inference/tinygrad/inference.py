from pathlib import Path
import json
import os
from exo.inference.tinygrad.models.llama import Transformer, convert_from_huggingface, fix_bf16
from exo.inference.shard import Shard
from exo.inference.tokenizers import resolve_tokenizer
from tinygrad.nn.state import safe_load, torch_load, load_state_dict
from tinygrad import Tensor, dtypes, nn, Context
from transformers import AutoTokenizer
from exo.inference.inference_engine import InferenceEngine
from typing import Optional, Tuple
import numpy as np
from exo.inference.tinygrad.tinygrad_helpers import concat_weights, load
from exo.download.shard_download import ShardDownloader
from concurrent.futures import ThreadPoolExecutor
import asyncio
import threading
from functools import partial

Tensor.no_grad = True
# default settings
TEMPERATURE = int(os.getenv("TEMPERATURE", 0.85))
TOP_K = 25
TOP_P = 0.9
ALPHA_F = 0.1
ALPHA_P = 0.0
MODEL_PARAMS = {
  "8B": {"args": {"dim": 4096, "n_heads": 32, "n_kv_heads": 8, "n_layers": 32, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256, "hidden_dim": 14336}, "files": 1},
  "70B": {"args": {"dim": 8192, "n_heads": 64, "n_kv_heads": 8, "n_layers": 80, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256, "hidden_dim": 28672}, "files": 8}
}


class MLXQuantizedLinear:
  def __init__(self, in_features, out_features, bits=4, group_size=64, bias=False):
    assert in_features % group_size == 0
    assert 32 % bits == 0
    assert (in_features * bits) % 32 == 0
    self.weight = Tensor.ones(out_features, (in_features * bits) // 32, dtype=dtypes.uint32)
    self.scales = Tensor.ones(out_features, in_features // group_size, dtype=dtypes.half)
    if bias:
      self.biases = Tensor.ones(out_features, in_features // group_size, dtype=dtypes.half)
    self.bits = bits
    self.group_size = group_size

  def __call__(self, x):
    M, K = x.shape
    N, K_packed = self.weight.shape

    num_values_per_uint32 = 32 // self.bits
    K_unpacked = K_packed * num_values_per_uint32
    num_groups = K // self.group_size
    packs_per_group = self.group_size // num_values_per_uint32

    assert K == K_unpacked, f"Mismatch in K dimensions: {K} vs {K_unpacked}"
    assert self.scales.shape == self.biases.shape == (N, num_groups), f"Scales must have shape (N, {num_groups}), got {self.scales.shape}"
    assert K % self.group_size == 0, "K must be divisible by the number of groups"

    bitmask = (1 << self.bits) - 1

    x_grouped = x.reshape(M, num_groups, self.group_size)

    output = Tensor.zeros((M, N), dtype=dtypes.float16)

    shift_list = [i * self.bits for i in range(num_values_per_uint32)]

    for g in range(num_groups):
        scale_g = self.scales[:, g].reshape(N, 1)
        bias_g = self.biases[:, g].reshape(N, 1)

        pack_start = g * packs_per_group
        pack_end = pack_start + packs_per_group
        w_packed_group = self.weight[:, pack_start:pack_end]

        unpacked_values = []

        for shift_amount in shift_list:
            shifted = w_packed_group >> shift_amount
            masked = (shifted & bitmask).cast(dtypes.float16)
            masked = masked.reshape(N, -1)

            unpacked_values.append(masked)

        w_unpacked_stack = Tensor.stack(*unpacked_values, dim=0)
        w_unpacked_group = w_unpacked_stack.permute(1, 2, 0).reshape(N, self.group_size)
        w_group = w_unpacked_group * scale_g + bias_g

        x_group = x_grouped[:, g, :]

        partial_output = x_group @ w_group.T
        output += partial_output

    return output
  
# class MLXQuantizedEmbedding:
#   def __init__(self, vocab_size, embed_size, bits = 4, group_size= 64):
#     self.vocab_sz, self.embed_sz = vocab_size, embed_size
#     self.bits = bits
#     self.group_size = group_size
#     self.weight = 
#     Tensor.glorot_uniform(vocab_size, embed_size)

def build_transformer(model_path: Path, shard: Shard, model_size="8B", device=None):
  try:
    with open(model_path/"config.json", "r") as f:
      config = json.load(f)
  except FileNotFoundError:
    raise Exception(f"Config file not found in {model_path}")
  
  # build model
  if (quantization := config.get("quantization", None)) is not None:
    linear = partial(MLXQuantizedLinear, **quantization)
  else:
    linear = nn.Linear
  with Context(THREEFRY=0):
    model = Transformer(**MODEL_PARAMS[model_size]["args"], linear=linear, max_context=8192, jit=True, shard=shard)

  # load weights
  if model_path.is_dir():
    if (model_path/"model.safetensors.index.json").exists(): weights = load(str(model_path/"model.safetensors.index.json"), shard)
    elif (model_path/"model.safetensors").exists(): weights = load(str(model_path/"model.safetensors"), shard)
    else: weights = concat_weights([load(str(model_path/f"consolidated.{i:02d}.pth"), shard) for i in range(MODEL_PARAMS[model_size]["files"])], device[0] if isinstance(device, tuple) else device)
  else:
    weights = load(str(model_path), shard)
  weights = convert_from_huggingface(weights, model, MODEL_PARAMS[model_size]["args"]["n_heads"], MODEL_PARAMS[model_size]["args"]["n_kv_heads"])
  weights = fix_bf16(weights)

  with Context(BEAM=0):
    # replace weights in model
    load_state_dict(model, weights, strict=False, consume=False)  # consume=True
  return model


class TinygradDynamicShardInferenceEngine(InferenceEngine):
  def __init__(self, shard_downloader: ShardDownloader):
    self.shard = None
    self.shard_downloader = shard_downloader
    self.executor = ThreadPoolExecutor(max_workers=1)

  async def infer_prompt(self, request_id: str, shard: Shard, prompt: str, image_str: Optional[str] = None, inference_state: Optional[str] = None) -> (np.ndarray, str, bool):
    await self.ensure_shard(shard)
    start_pos = json.loads(inference_state or "{}").get("start_pos", 0)
    n_captured_toks = json.loads(inference_state or "{}").get("n_captured_toks", 0)

    toks = await asyncio.get_event_loop().run_in_executor(self.executor, self.tokenizer.encode, prompt)
    h = await asyncio.get_event_loop().run_in_executor(self.executor, lambda: self.model(Tensor([toks]), start_pos, TEMPERATURE).realize())

    if h.shape == (1,):
      start_pos += len(toks)
      start_pos += 1
      n_captured_toks = 0
      return np.array([[h.item()]]), json.dumps({"start_pos": start_pos, "n_captured_toks": n_captured_toks}), h.item() == self.tokenizer.eos_token_id
    else:
      n_captured_toks = len(toks)
      return h.numpy(), json.dumps({"start_pos": start_pos, "n_captured_toks": n_captured_toks}), False

  async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[str] = None) -> Tuple[np.ndarray, str, bool]:
    await self.ensure_shard(shard)
    start_pos = json.loads(inference_state or "{}").get("start_pos", 0)
    n_captured_toks = json.loads(inference_state or "{}").get("n_captured_toks", 0)

    h = await asyncio.get_event_loop().run_in_executor(self.executor, lambda: self.model(Tensor(input_data), start_pos, TEMPERATURE).realize())

    if h.shape == (1,):
      start_pos += n_captured_toks
      start_pos += 1
      n_captured_toks = 0
      return np.array([[h.item()]]), json.dumps({"start_pos": start_pos, "n_captured_toks": n_captured_toks}), h.item() == self.tokenizer.eos_token_id
    else:
      return h.numpy(), json.dumps({"start_pos": start_pos, "n_captured_toks": n_captured_toks}), False

  async def ensure_shard(self, shard: Shard):
    if self.shard == shard:
      return

    model_path = await self.shard_downloader.ensure_shard(shard)

    if self.shard != shard:
      self.model = await asyncio.get_event_loop().run_in_executor(self.executor, build_transformer, model_path, shard, "8B" if "8b" in shard.model_id.lower() else "70B")

      tokenizer_path = str((model_path if model_path.is_dir() else model_path.parent))
      self.tokenizer = await resolve_tokenizer(tokenizer_path)
      self.shard = shard
