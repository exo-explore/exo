from pathlib import Path
import json
import os
from exo.inference.tinygrad.models.llama import Transformer, TransformerShard, convert_from_huggingface, fix_bf16, sample_logits
from exo.inference.shard import Shard
from exo.inference.tokenizers import resolve_tokenizer
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict
from tinygrad import Tensor, nn, Context, TinyJit
from exo.inference.inference_engine import InferenceEngine
import numpy as np
from exo.inference.tinygrad.tinygrad_helpers import concat_weights, load
from exo.download.shard_download import ShardDownloader
from concurrent.futures import ThreadPoolExecutor
from .stateful_model import make_prompt_state
from .losses import length_masked_ce_loss
from collections import OrderedDict
import asyncio
from typing import Optional
Tensor.no_grad = True 
# default settings
TEMPERATURE = int(os.getenv("TEMPERATURE", 0.85))
TOP_K = 25
TOP_P = 0.9
ALPHA_F = 0.1
ALPHA_P = 0.0
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
  "70B": {"args": {"dim": 8192, "n_heads": 64, "n_kv_heads": 8, "n_layers": 80, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256, "hidden_dim": 28672}, "files": 8}
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
  weights = convert_from_huggingface(weights, model, MODEL_PARAMS[model_size]["args"]["n_heads"], MODEL_PARAMS[model_size]["args"]["n_kv_heads"])
  weights = fix_bf16(weights)

  with Context(BEAM=0):
    # replace weights in model
    load_state_dict(model, weights, strict=False, consume=False)  # consume=True
    model = TransformerShard(shard, model)

  return model

_executor = ThreadPoolExecutor(max_workers=1) # singleton so tinygrad always runs on the same thread
class TinygradDynamicShardInferenceEngine(InferenceEngine):
  def __init__(self, shard_downloader: ShardDownloader):
    self.shard = None
    self.shard_downloader = shard_downloader
    self.states = OrderedDict()
    self.executor = _executor

  def poll_state(self, x, request_id: str, max_states=2):
    if request_id not in self.states:
      if len(self.states) >= max_states:
        self.states.popitem(last=False)
      self.states[request_id] = make_prompt_state(x, self.model)
    else:
      self.states.move_to_end(request_id)
    state = self.states[request_id]
    return {"start_pos": state.start, "cache": state.cache}

  async def sample(self, x: np.ndarray, temp=TEMPERATURE, top_p: float = 0.0) -> np.ndarray:
    def sample_wrapper():
      logits = x[:, -1, :]
      return sample_logits(Tensor(logits).flatten(), temp, 0, 0.8, top_p, 0.0).realize().numpy().astype(int)
    return await asyncio.get_running_loop().run_in_executor(self.executor, sample_wrapper)

  async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
    await self.ensure_shard(shard)
    tokens = await asyncio.get_running_loop().run_in_executor(self.executor, self.tokenizer.encode, prompt)
    return await asyncio.get_running_loop().run_in_executor(self.executor, np.array, tokens)
  
  async def decode(self, shard: Shard, tokens) -> str:
    await self.ensure_shard(shard)
    tokens = await asyncio.get_running_loop().run_in_executor(self.executor, self.tokenizer.decode, tokens)
    return tokens
  
  async def load_checkpoint(self, shard: Shard, path: str):
    await self.ensure_shard(shard)
    state_dict = safe_load(path)
    await asyncio.get_running_loop().run_in_executor(self.executor, load_state_dict, self.model, state_dict)
  
  async def save_checkpoint(self, shard: Shard, path: str):
    await self.ensure_shard(shard)
    state_dict = await asyncio.get_running_loop().run_in_executor(self.executor, get_state_dict, self.model)
    safe_save(state_dict, path) 
  
  async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict]]:
    await self.ensure_shard(shard)
    def wrap_infer():
      x = Tensor(input_data)
      h = self.model.embed(x)
      state = self.poll_state(h, request_id)
      out = self.model.forward(h, **state)
      self.states[request_id].start += x.shape[1]
      return out.numpy()
    output_data = await asyncio.get_running_loop().run_in_executor(self.executor, wrap_infer)
    return output_data, inference_state

  async def evaluate(self, request_id: str, shard: Shard, inputs, targets, lengths, loss=length_masked_ce_loss):
    def step(x, y, l):
      Tensor.training = False
      return self.session['loss'](self.model, x, y, l)
    await self.ensure_shard(shard)
    score = await asyncio.get_running_loop().run_in_executor(self.executor, lambda: self.session['jit'](Tensor(inputs), targets, lengths))
    out = score.numpy()
    return out
  
  async def train(self, request_id: str, shard: Shard, inputs, targets, lengths, loss=length_masked_ce_loss, opt=nn.optim.Adam, lr=1e-5):
    def step(x, y, l):
      Tensor.training = True
      score = self.session['loss'](self.model, x, y, l)
      self.session['opt'].zero_grad()
      score.backward()
      self.session['opt'].step()
      return score
    await self.ensure_shard(shard)
      
    score = await asyncio.get_running_loop().run_in_executor(self.executor, lambda: self.session['jit'](Tensor(inputs), targets, lengths).realize())
    
    return loss.numpy(), loss.numpy()

  async def ensure_shard(self, shard: Shard):
    if self.shard == shard:
      return

    model_path = await self.shard_downloader.ensure_shard(shard, self.__class__.__name__)

    if self.shard != shard:
      loop = asyncio.get_running_loop()
      parameters = "1B" if "1b" in shard.model_id.lower() else "3B" if "3b" in shard.model_id.lower() else "8B" if "8b" in shard.model_id.lower() else "70B"
      model_shard = await loop.run_in_executor(self.executor, build_transformer, model_path, shard, parameters)

      tokenizer_path = str((model_path if model_path.is_dir() else model_path.parent))
      self.tokenizer = await resolve_tokenizer(tokenizer_path)
      self.shard = shard
      self.model = model_shard
