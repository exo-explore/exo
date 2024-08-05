from functools import partial
from pathlib import Path
from typing import List, Optional, Union, Callable, Coroutine, Any
import json
from tiktoken.load import load_tiktoken_bpe
from exo.inference.tinygrad.models.llama import Transformer, convert_from_huggingface, fix_bf16
from tinygrad.nn.state import safe_load, torch_load, load_state_dict
from tinygrad import Tensor, nn, Context, GlobalCounters
from tinygrad.helpers import DEBUG, tqdm, _cache_dir, fetch
from exo.inference.shard import Shard
from exo.inference.inference_engine import InferenceEngine
import numpy as np
from exo.inference.hf_helpers import HFRepoProgressCallback, HFRepoProgressEvent, download_all_files, get_repo_root

MODEL_PARAMS = {
  "8B": {
    "args": {
      "dim": 4096,
      "n_heads": 32,
      "n_kv_heads": 8,
      "n_layers": 32,
      "norm_eps": 1e-5,
      "rope_theta": 500000,
      "vocab_size": 128256,
      "hidden_dim": 14336,
    },
    "files": 1,
  },
  "70B": {
    "args": {
      "dim": 8192,
      "n_heads": 64,
      "n_kv_heads": 8,
      "n_layers": 80,
      "norm_eps": 1e-5,
      "rope_theta": 500000,
      "vocab_size": 128256,
      "hidden_dim": 28672,
    },
    "files": 8,
  },
}



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


def load(fn: str):
  if fn.endswith(".index.json"):
    with open(fn) as fp:
      weight_map = json.load(fp)["weight_map"]
    parts = {n: load(str(Path(fn).parent / Path(n).name)) for n in set(weight_map.values())}
    return {k: parts[n][k] for k, n in weight_map.items()}
  elif fn.endswith(".safetensors"):
    return safe_load(fn)
  else:
    return torch_load(fn)


def build_transformer(model_path: Path, shard: Shard, model_size="8B", quantize=None, device=None):
  # build model
  linear = nn.Linear
  with Context(THREEFRY=0):
    model = Transformer(**MODEL_PARAMS[model_size]["args"], shard=shard, linear=linear, max_context=8192, jit=False)

  # load weights
  if model_path.is_dir():
    if (model_path / "model.safetensors.index.json").exists():
      weights = load(str(model_path / "model.safetensors.index.json"))
    elif (model_path / "model.safetensors").exists():
      weights = load(str(model_path / "model.safetensors"))
    else:
      weights = concat_weights(
        [load(str(model_path / f"consolidated.{i:02d}.pth")) for i in range(MODEL_PARAMS[model_size]["files"])],
        device[0] if isinstance(device, tuple) else device,
      )
  else:
    weights = load(str(model_path))
  if "model.embed_tokens.weight" in weights:
    weights = convert_from_huggingface(
      weights,
      model,
      MODEL_PARAMS[model_size]["args"]["n_heads"],
      MODEL_PARAMS[model_size]["args"]["n_kv_heads"],
      shard=shard,
    )
  weights = fix_bf16(weights)

  with Context(BEAM=0):
    # quantize
    if quantize is not None:
      weights = linear.quantize(weights, device)
      for _, v in weights.items():
        v.realize()

    # shard
    if isinstance(device, tuple):
      for k, v in nn.state.get_state_dict(model).items():
        if "scale" in k:
          v.shard_(device, axis=None)  # from quantized
        elif ".attention." in k:
          v.shard_(device, axis=-1)
        elif ".feed_forward.w1." in k:
          v.shard_(device, axis=0)
        elif ".feed_forward.w3." in k:
          v.shard_(device, axis=0)
        elif ".feed_forward." in k:
          v.shard_(device, axis=-1)
        elif "tok_embeddings.weight" in k:
          v.shard_(device, axis=0)
        elif "output.weight" in k:
          v.shard_(device, axis=0)
        else:
          v.shard_(device, axis=None)

    # replace weights in model
    load_state_dict(model, weights, strict=False, consume=True)
  return model


# default settings
TEMPERATURE = 0  # 0.85
TOP_K = 25
TOP_P = 0.9
ALPHA_F = 0.1
ALPHA_P = 0.0


def prefill(model, toks, start_pos=0):
  # prefill the model
  for tok in tqdm(toks):
    GlobalCounters.reset()
    model(Tensor([[tok]]), start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P).realize()
    start_pos += 1
  return start_pos


class TinygradDynamicShardInferenceEngine(InferenceEngine):
  def __init__(self, progress_callback: Optional[HFRepoProgressCallback] = None):
    self.shard = None
    self.progress_callback = progress_callback

  async def infer_prompt(self, request_id: str, shard: Shard, prompt: str, image_str: Optional[str] = None, inference_state: Optional[str] = None) -> (np.ndarray, str, bool):
    # TODO: we need to refactor models/llamaa to handle per-request-kv-cache. right now it's shared between requests.
    await self.ensure_shard(shard)
    start_pos = json.loads(inference_state).get("start_pos", 0) if inference_state else 0

    toks = self.tokenizer.encode(prompt)
    start_pos = prefill(self.model, toks[:-1], start_pos=start_pos)
    last_tok = toks[-1]

    output_data = np.array([self.model(Tensor([[last_tok]]), start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P).tolist()])
    if output_data.size == 1:
      start_pos += 1

    return (
      output_data,
      json.dumps({"start_pos": start_pos}),
      output_data.size == 1 and output_data.item() in [self.tokenizer.eos_token_id],
    )

  async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[str] = None) -> (np.ndarray, str, bool):
    await self.ensure_shard(shard)
    start_pos = json.loads(inference_state).get("start_pos", 0) if inference_state else 0

    output_data: np.ndarray = np.array([self.model(Tensor([input_data]), start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P).tolist()])
    if output_data.size == 1:
      start_pos += 1

    return (
      output_data,
      json.dumps({"start_pos": start_pos}),
      output_data.size == 1 and output_data.item() in [self.tokenizer.eos_token_id],
    )

  async def ensure_shard(self, shard: Shard):
    if self.shard == shard:
      return

    model_path = await download_all_files(shard.model_id, progress_callback=self.progress_callback)
    print(f"{model_path=}")
    model = build_transformer(model_path, shard=shard, model_size="8B" if "8b" in shard.model_id else "70B" if "70b" in shard.model_id else "8B")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str((model_path if model_path.is_dir() else model_path.parent)))

    self.shard = shard
    self.model = model
    self.tokenizer = tokenizer

  def set_progress_callback(self, progress_callback: Callable[[HFRepoProgressEvent], Coroutine[Any, Any, None]]):
    self.progress_callback = progress_callback