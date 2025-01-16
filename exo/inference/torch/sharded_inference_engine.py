"""
TorchDynamicShardInferenceEngine
Sharded inference engine using PyTorch based torchtune models
"""

import os
import functools
from concurrent.futures import ThreadPoolExecutor
import asyncio
import uuid
from typing import Optional

import numpy as np
import torch
from torchtune.generation import sample as tt_sample
from transformers import AutoTokenizer

from exo.inference.inference_engine import InferenceEngine
from exo.download.hf.hf_shard_download import HFShardDownloader
from exo.inference.shard import Shard
from exo.inference.tokenizers import _resolve_tokenizer
from exo.helpers import DEBUG
from exo.inference.torch.models.llm_utils import (
  load_model_config,
  load_model_weights_torchtune,
)

# supported models
from exo.inference.torch.models.llama3 import ShardedLlamaModel

TEMP = 0.0
TOP_K = 35


class TorchDynamicShardInferenceEngine(InferenceEngine):
  """
  Pytorch based inferece engine for sharded models
  """
  def __init__(self, shard_downloader: HFShardDownloader):
    self.shard = None
    self.shard_downloader = shard_downloader
    self.request_id = None
    self.executor = ThreadPoolExecutor(max_workers=1)
    self.past_tokens = None
    self.uuid = str(uuid.uuid4())

    # device settings
    if os.environ.get("TORCH_DEVICE"):
      self.device = torch.device(os.environ["TORCH_DEVICE"])
    elif torch.cuda.is_available():
      self.device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
      self.device = torch.device("mps")
    else:
      self.device = torch.device("cpu")

  async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
    if DEBUG >= 4:
      print("encode called")
      print(f"shard: {shard}")
      print(f"prompt: {prompt}")

    await self.ensure_shard(shard)

    tokens = await asyncio.get_event_loop().run_in_executor(
      self.executor,
      functools.partial(self.tokenizer.encode, prompt, return_tensors="np"),
    )

    if DEBUG >= 4:
      print(f"tokens: {tokens}")

    return tokens

  async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
    if DEBUG >= 4:
      print("decode called")
      print(f"shard: {shard}")
      print(f"tokens: {tokens}")

    await self.ensure_shard(shard)

    return await asyncio.get_running_loop().run_in_executor(
      self.executor,
      functools.partial(self.tokenizer.decode, tokens.tolist()),
    )

  async def sample(self, x: np.ndarray, temp=TEMP, top_k=TOP_K) -> np.ndarray:
    if DEBUG >= 4:
      print("sample called")
      print(f"x: {x}")
      print(f"temp: {temp}")
      print(f"top_k: {top_k}")

    logits = torch.tensor(x).to(self.device)

    def sample_wrapper():
      tokens = tt_sample(logits, temperature=temp, top_k=top_k)
      if DEBUG >= 4:
        print(f"tokens: {tokens}")

      return tokens.numpy(force=True)

    return await asyncio.get_running_loop().run_in_executor(self.executor, functools.partial(sample_wrapper))

  async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict]]:
    # ensure shard
    if DEBUG >= 4:
      print("infer_tensor called")
      print(f"shard: {shard}")
      print(f"input_data: {input_data}")
      print(f"inference_state: {inference_state}")

    await self.ensure_shard(shard)

    if inference_state.get("past_tokens") is not None:
      self.past_tokens = torch.tensor(inference_state["past_tokens"])

    self.request_id = request_id if not self.request_id else self.request_id

    hidden_state = None
    if input_data.ndim == 3:
      hidden_state = torch.tensor(input_data).to(self.device)
    elif input_data.ndim == 2:
      input_tensor = torch.tensor(input_data).to(self.device)

      if self.past_tokens is not None:
        self.past_tokens = torch.cat([self.past_tokens, input_tensor], dim=-1).to(self.device)
      else:
        self.past_tokens = input_tensor.clone()

    def infer_wrapper():
      if DEBUG >= 4:
        print("infer_wrapper called")
        print(f"self.past_tokens: {self.past_tokens}")
        print(f"hidden_state: {hidden_state}")

      if hidden_state is not None:
        model_hs, model_logits = self.sharded_model.generate(hidden_state=hidden_state)
      else:
        if not self.sharded_model.model.caches_are_enabled():
          model_hs, model_logits = self.sharded_model.generate(tokens=self.past_tokens)
        else:
          model_hs, model_logits = self.sharded_model.generate(tokens=input_tensor)

      curr_inference_state = {
        "past_tokens": self.past_tokens.numpy(force=True).tolist(),
      }

      if model_hs is not None:
        # model_hs = model_hs.detach().cpu()

        # possibly make this into a tensor that has past_tokens also
        # to pass to node, currently only hidden state is
        return model_hs.numpy(force=True), curr_inference_state

      # model_logits = model_logits.detach().cpu()
      # token = await self.sample(model_logits, TEMP, TOP_K)
      return model_logits[:, -1].numpy(force=True), curr_inference_state

    return await asyncio.get_running_loop().run_in_executor(self.executor, infer_wrapper)

  async def ensure_shard(self, shard: Shard):
    if DEBUG >= 4:
      print("shard ensured\n")
      print(f"shard: {shard}")
      print(f"class shard: {self.shard}")
      print(f"uuid: {self.uuid}")

    if self.shard == shard:
      return

    self.shard = shard

    # download model safetensors and shard
    model_path = await self.shard_downloader.ensure_shard(shard, self.__class__.__name__)
    model_config = load_model_config(model_path/"config.json")

    # self.tokenizer = await _resolve_tokenizer(model_path)
    self.tokenizer = await _resolve_tokenizer(model_path)
    eot_token = (
      self.tokenizer.special_tokens_map.get("eos_token_id")
      if hasattr(self.tokenizer, "_tokenizer") and isinstance(self.tokenizer._tokenizer, AutoTokenizer) else getattr(self.tokenizer, "eos_token_id", None)
    )

    print(f"eot_token: {eot_token}")
    print(self.tokenizer.special_tokens_map)
    print(self.tokenizer.eos_token_id)

    self.sharded_model = await asyncio.get_running_loop().run_in_executor(
      self.executor,
      functools.partial(
        ShardedLlamaModel,
        config=model_config,
        shard=shard,
        device=self.device,
        use_cache=os.environ.get("TORCH_USE_CACHE", True),
      ),
    )

    # load sharded weights
    await asyncio.get_running_loop().run_in_executor(
      self.executor,
      functools.partial(load_model_weights_torchtune, model_path, shard, self.sharded_model),
    )

  async def load_checkpoint(self, shard: Shard, path: str):
    await self.ensure_shard(shard)
