import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.sample_utils import make_sampler
import mlx.optimizers as optim
from ..inference_engine import InferenceEngine
from .sharded_utils import load_model_shard, resolve_tokenizer
from .losses import loss_fns
from ..shard import Shard
from typing import Dict, Optional, Tuple
from exo.download.shard_download import ShardDownloader
import asyncio
from collections import OrderedDict
from mlx_lm.models.cache import make_prompt_cache
from concurrent.futures import ThreadPoolExecutor

class MLXDynamicShardInferenceEngine(InferenceEngine):
  def __init__(self, shard_downloader: ShardDownloader):
    self.shard = None
    self.shard_downloader = shard_downloader
    self.caches = OrderedDict()
    self.sampler_params: tuple[float, float] = (0.0, 0.0, 0.0, 1)
    self.sampler = make_sampler(*self.sampler_params)
    self._mlx_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
    self._tokenizer_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tokenizer")
    self.session = {}
    self._shard_lock = asyncio.Lock()

  async def _eval_mlx(self, *args):
    await asyncio.get_running_loop().run_in_executor(self._mlx_thread, mx.eval, *args)

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
    result = self.sampler(logprobs)
    await self._eval_mlx(result)
    return np.asarray(result, dtype=int)

  async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
    await self.ensure_shard(shard)
    return np.asarray(
      await asyncio.get_running_loop().run_in_executor(
        self._tokenizer_thread,
        self.tokenizer.encode,
        prompt
      )
    )

  async def decode(self, shard: Shard, tokens) -> str:
    await self.ensure_shard(shard)
    return await asyncio.get_running_loop().run_in_executor(
      self._tokenizer_thread,
      self.tokenizer.decode,
      tokens
    )

  async def save_checkpoint(self, shard: Shard, path: str):
    await self.ensure_shard(shard)
    await asyncio.get_running_loop().run_in_executor(self._mlx_thread, lambda: self.model.save_weights(path))

  async def load_checkpoint(self, shard: Shard, path: str):
    await self.ensure_shard(shard)
    await asyncio.get_running_loop().run_in_executor(self._mlx_thread, lambda: self.model.load_weights(path))

  async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict]]:
    await self.ensure_shard(shard)
    state = await self.poll_state(request_id) if self.model.model_type != 'StableDiffusionPipeline' else {}
    x = mx.array(input_data)

    if self.model.model_type != 'StableDiffusionPipeline':
      output_data = await asyncio.get_running_loop().run_in_executor(
        self._mlx_thread,
        lambda: self.model(x, **state, **(inference_state or {}))
      )
      inference_state = None
    else:
      result = await asyncio.get_running_loop().run_in_executor(
        self._mlx_thread,
        lambda: self.model(x, **state, **(inference_state or {}))
      )
      output_data, inference_state = result

    await self._eval_mlx(output_data)
    output_data = await asyncio.get_running_loop().run_in_executor(
      self._mlx_thread,
      lambda: np.array(output_data, copy=False)
    )
    return output_data, inference_state

  async def evaluate(self, request_id: str, shard: Shard, inputs, targets, lengths, loss: str = "length_masked_ce"):
    await self.ensure_shard(shard)
    await self.save_session('loss', loss_fns[loss])
    x = mx.array(inputs)
    y = mx.array(targets)
    l = mx.array(lengths)

    score = await asyncio.get_running_loop().run_in_executor(
      self._mlx_thread,
      lambda: self.session['loss'](self.model, x, y, l)
    )
    return score

  async def ensure_train(self, shard: Shard, loss: str, opt=optim.SGD, lr=1e-5, trainable_layers=['input_layernorm', 'gate_proj']):
    await self.ensure_shard(shard)

    if 'train_layers' not in self.session or self.session['train_layers'] != trainable_layers:
      await self.save_session('train_layers', trainable_layers)
      def freeze_unfreeze():
        self.model.freeze()
        self.model.apply_to_modules(
          lambda k, v: v.unfreeze() if any(k.endswith(layer_name) for layer_name in trainable_layers) else None
        )
      await asyncio.get_running_loop().run_in_executor(self._mlx_thread, freeze_unfreeze)

    if 'lossname' not in self.session or 'LVaG' not in self.session or self.session['lossname'] != loss:
      await self.save_session('lossname', loss)
      await self.save_session('LVaG', nn.value_and_grad(self.model, loss_fns[loss]))

    if 'opt' not in self.session:
      await self.save_session('opt', opt(lr))
    return True

  async def train(self, request_id: str, shard: Shard, inputs, targets, lengths, loss: str = "length_masked_ce", opt=optim.SGD, lr=1e-5):
    await self.ensure_train(shard, loss, opt, lr)

    def train_step(inp, tar, lng):
      lval, grad = self.session['LVaG'](self.model, inp, tar, lng)
      gradlayers = grad['model']['layers']
      self.session['opt'].update(self.model, grad)
      return lval, gradlayers, (self.model.parameters(), self.session['opt'].state, lval)

    x = mx.array(inputs)
    y = mx.array(targets)
    l = mx.array(lengths)
    score, gradients, eval_args = await asyncio.get_running_loop().run_in_executor(
      self._mlx_thread,
      lambda: train_step(x, y, l)
    )
    await self._eval_mlx(*eval_args)

    layers = [{k: v["weight"] for k, v in layer.items() if 'weight' in v} for layer in gradients if layer]
    first_layer = np.array(layers[0]['input_layernorm'], copy=False)
    await self._eval_mlx(first_layer)
    return score, first_layer

  async def ensure_shard(self, shard: Shard):
    async with self._shard_lock:
      if self.shard == shard: return
      model_path = await self.shard_downloader.ensure_shard(shard, self.__class__.__name__)
      if self.shard != shard:
        model_shard = await asyncio.get_running_loop().run_in_executor(
          self._mlx_thread,
          lambda: load_model_shard(model_path, shard, lazy=False)
        )
        if hasattr(model_shard, "tokenizer"):
          self.tokenizer = model_shard.tokenizer
        else:
          self.tokenizer = await resolve_tokenizer(model_path)
        self.shard = shard
        self.model = model_shard
        self.caches = OrderedDict()
        self.session = {}

  async def cleanup(self):
    self._mlx_thread.shutdown(wait=True)
