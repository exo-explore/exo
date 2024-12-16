import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.sample_utils import top_p_sampling, make_sampler
import mlx.optimizers as optim
from ..inference_engine import InferenceEngine
from .sharded_utils import load_shard, get_image_from_str
from .losses import loss_fns 
from ..shard import Shard
from typing import Dict, Optional, Tuple
from exo.download.shard_download import ShardDownloader
import asyncio
from collections import OrderedDict
from mlx_lm.models.cache import make_prompt_cache

def sample_logits(
  logits: mx.array,
  temp: float = 0.0,
  top_p: float = 1.0,
  logit_bias: Optional[Dict[int, float]] = None
) -> Tuple[mx.array, float]:
  if logit_bias:
    indices = mx.array(list(logit_bias.keys()))
    values = mx.array(list(logit_bias.values()))
    logits[:, indices] += values

  if temp == 0:
    token = mx.argmax(logits, axis=-1)
  else:
    if top_p > 0 and top_p < 1.0:
      token = top_p_sampling(logits, top_p, temp)
    else:
      token = mx.random.categorical(logits*(1/temp))

  return token

class MLXDynamicShardInferenceEngine(InferenceEngine):
  def __init__(self, shard_downloader: ShardDownloader):
    self.shard = None
    self.shard_downloader = shard_downloader
    self.caches = OrderedDict()
    self.sampler_params: tuple[float, float] = (0.0, 0.0, 0.0, 1)
    self.sampler = make_sampler(*self.sampler_params)

  async def poll_state(self, request_id: str, max_caches=2):
    if request_id in self.caches:
      self.caches.move_to_end(request_id)
    else:
      newcache = make_prompt_cache(self.model)
      if len(self.caches) > max_caches:
        self.caches.popitem(last=False)
      self.caches[request_id] = newcache
    return {"cache": self.caches[request_id]}

  async def sample(self, x: np.ndarray, temp: float = 0.0, top_p: float = 1.0) -> np.ndarray:
    if (temp, top_p, 0.0, 1) != self.sampler_params:
      self.sampler_params = (temp, top_p, 0.0, 1)
      self.sampler = make_sampler(*self.sampler_params)
    logits = mx.array(x)
    logits = logits[:, -1, :]
    logprobs = logits - mx.logsumexp(logits, keepdims=True)
    return np.asarray(self.sampler(logprobs), dtype=int)

  async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
    await self.ensure_shard(shard)
    tokens = self.tokenizer.encode(prompt)
    return np.asarray(tokens)

  async def decode(self, shard: Shard, tokens) -> str:
    await self.ensure_shard(shard)
    return self.tokenizer.decode(tokens)

  async def save_checkpoint(self, shard: Shard, path: str):
    await self.ensure_shard(shard)
    self.model.save_weights(path)

  async def load_checkpoint(self, shard: Shard, path: str):
    await self.ensure_shard(shard)
    self.model.load_weights(path)
    
  async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray) -> np.ndarray:
    await self.ensure_shard(shard)
    state = await self.poll_state(request_id)
    x = mx.array(input_data)
    output_data = np.array(self.model(x, **state), copy=False)
    return output_data

  async def evaluate(self, request_id: str, shard: Shard, inputs, targets, lengths, loss: str = "length_masked_ce"):
    await self.ensure_shard(shard)
    await self.save_session('loss', loss_fns[loss])
    x = mx.array(inputs)
    y = mx.array(targets)
    l = mx.array(lengths)
    score = self.session['loss'](self.model, x, y, l)
    return score

  async def ensure_train(self, shard: Shard, loss: str, opt=optim.SGD, lr=1e-5, trainable_layers=['input_layernorm', 'gate_proj']):
    await self.ensure_shard(shard)
    if 'train_layers' not in self.session or self.session['train_layers'] != trainable_layers:
      await self.save_session('train_layers', trainable_layers)
      self.model.freeze()
      self.model.apply_to_modules(lambda k, v: v.unfreeze() if any(lambda: k.endswith(i) for i in trainable_layers) else None)
    if 'lossname' not in self.session or 'LVaG' not in self.session or self.session['lossname'] != loss:
      await self.save_session('lossname', loss)
      await self.save_session('LVaG', nn.value_and_grad(self.model, loss_fns[loss]))
    if 'opt' not in self.session:
      await self.save_session('opt', opt(lr))
    return True

  async def train(self, request_id: str, shard: Shard, inputs, targets, lengths, loss: str = "length_masked_ce", opt=optim.SGD, lr=1e-5):
    loop = asyncio.get_running_loop()
    nothin = await self.ensure_train(shard, loss, opt, lr)
    def train_step(inp, tar, lng):
      lval, grad = self.session['LVaG'](self.model, inp, tar, lng)
      gradlayers = grad['model']['layers']
      self.session['opt'].update(self.model, grad)
      mx.eval(self.model.parameters(), self.session['opt'].state, lval)
      return lval, gradlayers

    x = mx.array(inputs)
    y = mx.array(targets)
    l = mx.array(lengths)

    score, gradients = await loop.run_in_executor(self.executor, train_step, x, y, l)
    #print(f"{score=}")
      
    layers = [{k: v["weight"] for k,v in l.items() if 'weight' in v} for l in gradients if l]
    #print(layers[0])

    return score, np.array(layers[0]['input_layernorm'], copy=False)

  async def ensure_shard(self, shard: Shard):
    if self.shard == shard:
      return

    model_path = await self.shard_downloader.ensure_shard(shard, self.__class__.__name__)

    if self.shard != shard:
      model_shard, self.tokenizer = await load_shard(model_path, shard)
      self.shard = shard
      self.model = model_shard 
      self.caches = OrderedDict()
      self.session = {}

