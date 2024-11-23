"""
TorchDynamicShardInferenceEngine
Sharded inference engine using PyTorch based torchtune models
"""
import os
from typing import Optional, Tuple, Union, List
import functools
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import asyncio
import torch

from torchtune.models import llama3

from exo.inference.inference_engine import InferenceEngine
from exo.download.hf.hf_shard_download import HFShardDownloader
from exo.inference.shard import Shard
from exo.helpers import DEBUG
from exo.inference.torch.models.llm_utils import (
  load_model_config,
  load_model_weights_torchtune,
)

# supported models
from exo.inference.torch.models.llama3 import ShardedLlamaModel

TEMP = 0.6
TOP_K = 25

class TorchDynamicShardInferenceEngine(InferenceEngine):
  def __init__(self, shard_downloader: HFShardDownloader, model_id: str="llama"):
    self.shard = None
    self.shard_downloader = shard_downloader
    self.model_id = model_id
    self.supported_models = ["llama"]

    # device settings
    if os.environ.get("TORCH_DEVICE"):
      self.device = torch.device(os.environ["TORCH_DEVICE"])
    elif torch.cuda.is_available():
      self.device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
      self.device = torch.device("mps")
    else:
      self.device = torch.device("cpu")

  async def infer_prompt(
    self,
    request_id: str,
    shard: Shard,
    prompt: str,
    image_str: Optional[str] = None,
    inference_state: Optional[str] = None
  ) -> Tuple[np.ndarray, str, bool]:
    if DEBUG >= 4:
      print("infer_prompt called")
      print(f"prompt: {prompt}")
      print(f"shard: {shard}")
      print(f"inference_state: {inference_state}")
    # ensure shard
    await self.ensure_shard(shard)

    # tokenize
    tokens = torch.tensor(
      self.tokenizer.encode(prompt, add_bos=True, add_eos=True),
      dtype=torch.int
    )
    hidden_states = None

    # generate
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
      hidden_states, logits, finished = await loop.run_in_executor(
        pool,
        functools.partial(
          self.sharded_model.generate,
          tokens=tokens
        )
      )

    if hidden_states is not None:
      return hidden_states.numpy(force=True), "", finished
    else:
      return logits.numpy(force=True), "", finished

  async def infer_tensor(
    self,
    request_id: str,
    shard: Shard,
    input_data: np.ndarray,
    inference_state: Optional[str] = None
  ) -> Tuple[np.ndarray, str, bool]:
    # ensure shard
    await self.ensure_shard(shard)

    return np.empty((1,1)), "", False

  async def ensure_shard(self, shard: Shard):
    if self.shard == shard:
      return
    
    # download model safetensors and shard
    model_path = await self.shard_downloader.ensure_shard(shard)
    model_config = load_model_config(model_path / "config.json")

    self.tokenizer = llama3.llama3_tokenizer(
      path=f"{model_path}/original/tokenizer.model"
    )

    if self.model_id not in self.supported_models:
      raise ValueError(
        f"Model {self.model_id} not supported, only supported models are\n{self.supported_models}"
      )
    
    self.sharded_model = ShardedLlamaModel(
      model_config,
      shard,
      self.tokenizer,
      self.device,
      None,
      use_cache=True
    )

    # load sharded weights
    load_model_weights_torchtune(
      model_path,
      shard,
      self.sharded_model
    )
