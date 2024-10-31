from pathlib import Path
import json
import os
from exo.inference.tinygrad.models.llama import Transformer, convert_from_huggingface, fix_bf16
from exo.inference.shard import Shard
from exo.inference.tokenizers import resolve_tokenizer
from tinygrad.nn.state import load_state_dict
from tinygrad import Tensor, nn, Context, dtypes
from exo.inference.inference_engine import InferenceEngine
from typing import Optional, Tuple
import numpy as np
from exo.inference.tinygrad.tinygrad_helpers import concat_weights, load
from exo.download.shard_download import ShardDownloader
from concurrent.futures import ThreadPoolExecutor
import asyncio

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


# **** quantized linears ****
class Int8Linear:
  def __init__(self, in_features, out_features, bias=False):
    assert bias == False
    self.weight = Tensor.ones(out_features, in_features, dtype=dtypes.int8)
    self.scale = Tensor.ones(out_features, dtype=dtypes.half)

  def __call__(self, x):
    return x.dot(self.weight.cast(dtype=dtypes.half).T*self.scale)

  @staticmethod
  def quantize(tensors, device):
    new_tensors = {}
    for name,v in tensors.items():
      if "feed_forward" in name or "attention.w" in name:
        assert "weight" in name, name
        scale = v.abs().max(axis=1) / 127.0
        int8_weight = (v.T/scale).T.cast(dtype=dtypes.int8)
        new_tensors[name] = int8_weight
        new_tensors[name.replace('weight', 'scale')] = scale
        if isinstance(device, tuple):
          new_tensors[name].shard_(device, axis=-1)
          new_tensors[name.replace('weight', 'scale')].shard_(device, axis=None)
      else:
        new_tensors[name] = v
    return new_tensors

def NF4Linear(block_size):
  _CODE = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
  ]
  CODE = Tensor.stack(*[Tensor(c, dtype=dtypes.float16) for c in _CODE])
  class _NF4Linear:
    def __init__(self, in_features, out_features, bias=False):
      assert not bias, "bias not supported"
      self.in_features, self.out_features = in_features, out_features
      self.weight = Tensor.empty(int(out_features * in_features / 2), dtype=dtypes.uint8)
      self.scale = Tensor.empty(int(out_features * in_features / block_size), 1, dtype=dtypes.float16)

    def __call__(self, x: Tensor) -> Tensor:
      high_bits = self.weight
      low_bits = (self.weight * 2 ** 4).contiguous()
      unpacked = Tensor.stack(high_bits, low_bits, dim=-1) // (2 ** 4)
      unscaled = CODE[unpacked].to(x.device).reshape(-1, block_size) * self.scale
      return x.linear(unscaled.reshape(self.out_features, self.in_features).T)

    @staticmethod
    def quantize(state_dict: dict[str, Tensor], device) -> dict[str, Tensor]:
      new_state_dict = {}
      for k, v in state_dict.items():
        if "feed_forward" in k or "attention.w" in k:
          grouped = v.reshape(-1, block_size)
          scale = (grouped.abs().max(axis=1, keepdim=True))
          coded = ((grouped / scale).unsqueeze(-1) - CODE.to(v.device)).abs().argmin(axis=-1).cast(dtypes.uint8).flatten()
          new_state_dict[k] = coded[::2] * 2 ** 4 + coded[1::2]
          new_state_dict[k.replace(".weight", ".scale")] = scale.cast(dtypes.float16)
          if isinstance(device, tuple):
            new_state_dict[k].shard_(device, axis=-1)
            new_state_dict[k.replace('weight', 'scale')].shard_(device, axis=None)
        else:
          new_state_dict[k] = v
      return new_state_dict
  return _NF4Linear


def build_transformer(model_path: Path, shard: Shard, model_size="8B", device=None, quantize="nf4"):
  # build model

  if quantize == "int8": linear = Int8Linear
  elif quantize == "nf4": linear = NF4Linear(64)
  else: linear = nn.Linear

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
    # quantize
    if quantize is not None:
      weights = linear.quantize(weights, device)
      for _,v in weights.items(): v.realize()
    load_state_dict(model, weights, strict=False, consume=False)  # consume=True
  return model


class TinygradDynamicShardInferenceEngine(InferenceEngine):
  def __init__(self, shard_downloader: ShardDownloader, quantize=None):
    self.shard = None
    self.shard_downloader = shard_downloader
    self.executor = ThreadPoolExecutor(max_workers=1)
    self.quantize = quantize

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
      self.model = await asyncio.get_event_loop().run_in_executor(self.executor, build_transformer, model_path, shard, "8B" if "8b" in shard.model_id.lower() else "70B", self.quantize)

      tokenizer_path = str((model_path if model_path.is_dir() else model_path.parent))
      self.tokenizer = await resolve_tokenizer(tokenizer_path)
      self.shard = shard