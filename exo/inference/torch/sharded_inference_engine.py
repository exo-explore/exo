"""
TorchDynamicShardInferenceEngine
Sharded inference engine using PyTorch based torchtune models
"""

import os
import functools
from concurrent.futures import ThreadPoolExecutor
import asyncio
import uuid
import re
from typing import Optional

import numpy as np
import torch
import torchtune.generation as ttg
from transformers import AutoTokenizer

from exo.inference.inference_engine import InferenceEngine
from exo.download.hf.hf_shard_download import HFShardDownloader
from exo.inference.shard import Shard
from exo.inference.tokenizers import _resolve_tokenizer
from exo.helpers import DEBUG
from exo.inference.torch.models.llm_utils import (
  load_model_config,
  load_weights_torch,
  ShardInferenceState
)

# supported models
from exo.inference.torch.models.llama3 import ShardedLlamaModel

# from torchtune generate recipe
# https://github.com/pytorch/torchtune/blob/main/recipes/configs/generation.yaml#L40
TEMP = 0.6
TOP_K = 35

class TorchDynamicShardInferenceEngine(InferenceEngine):
  """
  Pytorch based inferece engine for sharded models
  """
  def __init__(self, shard_downloader: HFShardDownloader):
    self.shard = None
    self.shard_downloader = shard_downloader
    self.sharded_model = None
    self.request_id = None
    self.executor = ThreadPoolExecutor(max_workers=1)
    self.uuid = str(uuid.uuid4())
    self.model_path = None
    self.model_config = None
    self.state = None
    self.oom_cnt = 0

    # device settings
    if os.environ.get("TORCH_DEVICE"):
      self.device = torch.device(os.environ["TORCH_DEVICE"])
    elif torch.cuda.is_available():
      self.device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
      self.device = torch.device("mps")
    else:
      self.device = torch.device("cpu")

    # rng setup for sampling
    self.rng = torch.Generator(device=self.device)
    self.rng.manual_seed(1234)

  def clear_model(self):
    """
    Clear out model and shard
    A way to avoid OOM issues
    
    All prompts are stored in VRAM
    while inference engine is up and using the same
    model class instance, this will clear it for each prompt.

    OOM issue might occur in longer chats/contexts depending on your machine.
    """
    if self.sharded_model.model.caches_are_enabled():
      self.sharded_model.model.reset_caches()
    
    del self.sharded_model
    self.sharded_model = None
    
    if self.device == torch.device("cuda"):
      torch.cuda.empty_cache()
    
    self.shard = None
    self.state = None

  async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
    if DEBUG >= 4:
      print("encode called")
      print(f"shard: {shard}")
      print(f"prompt: {prompt}")

    if self.sharded_model is not None:
      print("CLEARING SHARD AND MODEL - ENCODING")
      self.clear_model()

    await self.ensure_shard(shard)

    def encode_wrapper() -> np.ndarray:
      """
      Encode the tensors from prompt along with the
      initial input_pos and mask
      """
      tokens = self.tokenizer.encode(
        prompt,
        return_tensors="pt"
      )

      # move to proper device, default is CPU
      if tokens.device != self.device:
        tokens = tokens.to(device=self.device)
      
      if DEBUG >= 4:
        print("encoded_wrapper called")
        print(f"tokens: {tokens}")

      # if going past max, just take from max onward
      if len(tokens) > self.sharded_model.max_generated_tokens:
        max_gen_tokens = self.sharded_model.max_generated_tokens
        tokens = tokens[-max_gen_tokens:]

      self.state.tokens = tokens

      bsz, tklng = tokens.size()
      total_response_length = tklng + self.sharded_model.max_generated_tokens

      # setup cache
      if not self.sharded_model.model.caches_are_enabled():
        with self.device:
          self.sharded_model.model.setup_caches(
            bsz,
            self.model_config["torch_dtype"],
            decoder_max_seq_len=total_response_length
          )
      
      # setup max sequence length
      if not self.sharded_model.model.caches_are_enabled():
        max_seq_len = total_response_length
      else:
        max_seq_len = self.sharded_model.model.decoder_max_cache_seq_len

      # set pad_id
      if hasattr(self.tokenizer, "pad_id"):
        pad_id = self.tokenizer.pad_id
      elif hasattr(self.tokenizer, "pad_token_id"):
        print(f"pad_token_id: {self.tokenizer.pad_token_id}")
        if self.tokenizer.pad_token_id is not None:
          pad_id = self.tokenizer.pad_token_id
        else:
          pad_id = 0
      else:
        pad_id = 0
      
      padding_masks = tokens != pad_id
      if not padding_masks.all():
        padding_masks = torch.nn.functional.pad(
          padding_masks,
          (0, self.sharded_model.max_generated_tokens),
          value=True,
        )

        self.state.mask = ttg.get_causal_mask_from_padding_mask(padding_masks, target_seq_len=max_seq_len)

        self.state.input_pos = ttg.get_position_ids_from_padding_mask(padding_masks)
      else:
        self.state.mask = torch.tril(torch.ones(
          total_response_length,
          max_seq_len,
          dtype=torch.bool,
          device=self.device,
        )).unsqueeze(0)

        self.state.input_pos = torch.arange(0, total_response_length, device=self.device).unsqueeze(0)

      return tokens

    return await asyncio.get_running_loop().run_in_executor(
      self.executor,
      functools.partial(encode_wrapper),
    )

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
      print(self.device)

    logits = torch.tensor(x).to(self.device)

    def sample_wrapper():
      q = torch.empty((logits.size(0), self.sharded_model.model.tok_embeddings.num_embeddings), device=logits.device).exponential_(1, generator=self.rng)

      tokens = ttg.sample(logits.clone(), temperature=temp, top_k=top_k, q=q.to(self.device))
      
      if DEBUG >= 4:
        print(f"tokens: {tokens}")

      return tokens.numpy(force=True)

    return await asyncio.get_running_loop().run_in_executor(self.executor, functools.partial(sample_wrapper))

  async def infer_tensor(
    self,
    request_id: str,
    shard: Shard,
    input_data: np.ndarray,
    inference_state: Optional[dict] = None
  ) -> tuple[np.ndarray, Optional[dict]]:

    await self.ensure_shard(shard)

    # ensure shard
    if DEBUG >= 4:
      print("infer_tensor called")
      print(f"shard: {shard}")
      print(f"input_data: {input_data}")

    if inference_state.get("tokens") is not None:
      self.state.from_dict(inference_state)

    self.request_id = request_id if not self.request_id else self.request_id

    hidden_state = None
    if input_data.ndim == 3:
      hidden_state = torch.tensor(input_data).to(
        device=self.device,
        dtype=self.model_config["torch_dtype"]
      )
    elif input_data.ndim == 2:
      input_tensor = torch.tensor(input_data).to(
        device=self.device
      )
      
      if self.state.tokens is not None:
        self.state.tokens = torch.cat([
          self.state.tokens.to(self.device),
          input_tensor
        ], dim=-1).to(self.device)
      else:
        self.state.tokens = input_tensor.clone()

    def infer_wrapper():
      if DEBUG >= 4:
        print(f"infer_wrapper called [{self.oom_cnt} OOM]")
        print(f"self.state.tokens: {self.state.tokens}")
        print(f"hidden_state: {hidden_state}")

      model_cache = self.sharded_model.model.caches_are_enabled()

      try:
        in_tokens = self.state.tokens.clone().to(
          device=self.device
        )

        in_input_pos = self.state.input_pos.clone().to(
          device=self.device
        )

        in_mask = self.state.mask.clone().to(
          device=self.device
        )

        if hidden_state is not None:
          model_hs, model_logits = self.sharded_model.generate(
            hidden_state=hidden_state,
            curr_pos=self.state.curr_pos
          )
        else:
          if not model_cache:
            model_hs, model_logits = self.sharded_model.generate(
              tokens=in_tokens,
              input_pos=in_input_pos,
              mask=in_mask,
              curr_pos=self.state.curr_pos
            )
          else:
            model_hs, model_logits = self.sharded_model.generate(
              tokens=input_tensor,
              input_pos=in_input_pos,
              mask=in_mask,
              curr_pos=self.state.curr_pos
            )
      except torch.cuda.OutOfMemoryError:
        print(f"OOM on cuda, clearing model and stopping")
        self.oom_cnt += 1
        self.clear_model()
        return
      except Exception as err:
        print(f"infer_tensor err\n{err}")
        raise

      if model_hs is not None:
        # numpy current no support for bf16
        if model_hs.dtype == torch.bfloat16:
          model_hs = model_hs.float()

        if DEBUG >= 4:
          print("sending hidden states")
          print(f"model_hs: {model_hs.size()}")
          print(f"state.tokens: {self.state.tokens}")
          print(f"state.input_pos: {self.state.input_pos.size()}")
          print(f"state.mask: {self.state.mask.size()}")
        
        return (
          model_hs.numpy(force=True),
          self.state.to_dict(),
        )
      
      if self.state.curr_pos == 0:
        self.state.curr_pos = self.state.tokens.size(-1)
      else:
        self.state.curr_pos += 1

      # numpy current no support for bf16
      if model_logits.dtype == torch.bfloat16:
        model_logits = model_logits.float()

      return (
        model_logits[:, -1].numpy(force=True),
        self.state.to_dict(),
      )

    return await asyncio.get_running_loop().run_in_executor(self.executor, infer_wrapper)

  async def ensure_shard(self, shard: Shard):
    if DEBUG >= 4:
      print("shard ensured\n")
      print(f"shard: {shard}")
      print(f"class shard: {self.shard}")
      print(f"uuid: {self.uuid}")

    # reset model after last layer to fix OOM
    if self.shard == shard:
      return

    self.shard = shard

    # Using CPU to store inference state
    self.state = ShardInferenceState()

    # download model safetensors and shard

    self.model_path = await self.shard_downloader.ensure_shard(shard, self.__class__.__name__)
    self.model_config = load_model_config(self.model_path/"config.json")

    # self.tokenizer = await _resolve_tokenizer(model_path)
    self.tokenizer = await _resolve_tokenizer(self.model_path)
    
    self.sharded_model = await asyncio.get_running_loop().run_in_executor(
      self.executor,
      functools.partial(
        ShardedLlamaModel,
        config=self.model_config,
        shard=shard,
        device=self.device,
        dtype=self.model_config["torch_dtype"],
        use_cache=bool(os.getenv("TORCH_USE_CACHE", "True").lower() == "true"),
      ),
    )

    # load sharded weights
    await asyncio.get_running_loop().run_in_executor(
      self.executor,
      functools.partial(load_weights_torch, self.model_path, self.sharded_model.model, self.model_config),
    )

  async def load_checkpoint(self, shard: Shard, path: str):
    await self.ensure_shard(shard)
