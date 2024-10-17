# experimental, based off of tinygrad/inference.py
import asyncio
import os
import json
import functools
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import torch

from typing import Optional, Tuple, Union, List
from exo.inference.shard import Shard
from exo.inference.inference_engine import InferenceEngine
from exo.inference.torch.model.hf import ShardedHuggingFaceModel
from exo.inference.tokenizers import resolve_tokenizer
from exo.helpers import DEBUG
from exo.download.hf.hf_shard_download import HFShardDownloader
from exo.download.hf.hf_helpers import get_weight_map

from transformers import Cache

# model value options
TOP_K = 20
TEMP = 0.6
TOP_P = 0.9

class TorchDynamicShardInferenceEngine(InferenceEngine):
  """
  Torch Dynamic Shard Inference Engine for performing model inference with sharded Pytorch/HF based models.
  """

  def __init__(self, shard_downloader: HFShardDownloader):
    """
    Initialize the inference engine.

    Args:
      shard_downloader: Model and weights sharding download
    """
    self.shard = None
    self.shard_downloader = shard_downloader

    # the whole history with new logits need to
    # be passed to the model to reach the end token
    # even with caching
    self.past_input_ids = None

    # setup cuda device
    if os.environ.get("TORCH_DEVICE"):
      self.device = torch.device(os.environ["TORCH_DEVICE"])
    elif torch.cuda.is_available():
      self.device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
      self.device = torch.device("mps")
    else:
      self.device = torch.device("cpu")

    torch.set_default_device(self.device)

    # setup cude dtype
    self.dtype = torch.get_default_dtype()

    # setup device_map
    if os.environ.get("TORCH_DEVICE_MAP"):
      self.device_map = os.environ["TORCH_DEVICE_MAP"]
    else:
      self.device_map = str(self.device)

  def infer_caching(
    self,
    inference_state: Optional[str] = None
  ) -> Tuple[Optional[torch.Tensor], Optional[dict]]:
    """
    inference caching from inference_state json
    """
    # setup cache and cached input_ids
    past_iids = None
    cached_iids = None
    if inference_state is not None:
      try:
        infer_state = json.loads(inference_state)
      except ValueError:
        infer_state = None

      if infer_state is not None:
        cached_iids = infer_state["cached_iids"]
        if cached_iids is not None:
          past_iids = None
          if len(cached_iids) > 0:
            past_iids = torch.tensor(cached_iids["input_ids"]).to(self.device)
            cached_iids = {"input_ids": past_iids.tolist()}

      if DEBUG >= 4:
        print(f"cached_iids len: {len(cached_iids)}")
        print(f"cached_iids: {cached_iids}")

    return (past_iids, cached_iids)

  async def async_forward(
    self,
    input_ids: Optional[torch.Tensor] = None,
    hidden_states: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None
  ) -> Tuple[Optional[torch.Tensor], Optional[Union[Cache, List[torch.FloatTensor]]], Optional[torch.Tensor]]:
    """
    Asynchronously performs the forward pass using a stateful sharded model.

    Args:
        input_ids (torch.Tensor, optional): Input token IDs for the model. If not provided, `hidden_states` must be used.
        hidden_states (torch.Tensor, optional): Precomputed hidden states to be used instead of `input_ids`.
        attention_mask (torch.Tensor, optional): Mask to prevent attention on padding token indices.

    Returns:
        A tuple containing:

        - shard_hidden_states (torch.Tensor, optional): Hidden states resulting from the forward pass.
        - shard_past_kvs (list(torch.FloatTensor), optional): List of past key-value tensors (cache) used in the model.
        - shard_logits (torch.Tensor, optional): The logits computed during the forward pass.
    """
    loop = asyncio.get_running_loop()

    with ThreadPoolExecutor() as pool:
      result = await loop.run_in_executor(pool, functools.partial(
        self.stateful_sharded_model.forward,
        input_ids=input_ids,
        hidden_states=hidden_states,
        attention_mask=attention_mask
      ))

    if DEBUG >=4:
      print("async_forward")
      print(f"result: {result}")

    return result[0], result[1], result[2]

  async def async_logit_sample(
    self,
    logits: torch.Tensor
  ) -> torch.Tensor:
    """
    Asynchronously samples logits using the model's logit sampling method.

    Args:
        logits (torch.Tensor): The logits produced by the model for sampling.

    Returns:
        next_logit (torch.Tensor): The next logit samples from given logis
    """
    loop = asyncio.get_running_loop()

    with ThreadPoolExecutor() as pool:
      result = await loop.run_in_executor(pool, functools.partial(
        self.stateful_sharded_model.logits_sample,
        logits=logits
      ))

    return result

  async def infer_prompt(
    self,
    request_id: str,
    shard: Shard,
    prompt: str,
    image_str: Optional[str] = None,
    inference_state: Optional[str] = None
  ) -> Tuple[np.ndarray, str, bool]:
    """
    Asynchronously processes a prompt using the specified shard and returns the inference result.

    Args:
        request_id (str): The unique identifier for the request.
        shard (Shard): The model shard used for inference.
        prompt (str): The text prompt to be processed by the model.
        image_str (str, optional): A base64 encoded image string to be optionally used in the inference. Defaults to None.
        inference_state (str, optional): The cached inference state for resuming or continuing inference. Defaults to None.

    Returns:
        A tuple containing:

        - input_ids (np.ndarray): The processed token IDs as a NumPy array if logits were generated. Otherwise, it returns hidden states.
        - cache_json (str): A JSON string containing the cached input IDs for further inference steps.
        - is_finished (bool): A boolean indicating whether the model has reached the end-of-sequence (EOS) token.
    """
    if DEBUG >= 4:
      print("infer_prompt called")
      print(f"prompt: {prompt}")
      print(f"shard: {shard}")
      print(f"inference_state: {inference_state}")

    await self.ensure_shard(shard)

    inputs = self.tokenizer([prompt], return_tensors="pt")
    input_ids = inputs.input_ids.to(self.device)
    input_attention_mask = inputs.attention_mask.to(self.device)

    # get cache from inference_state
    past_iids, cached_iids = self.infer_caching(inference_state)

    if past_iids is not None:
      self.past_input_ids = past_iids
    else:
      self.past_input_ids = input_ids

    if DEBUG >= 4:
      print(f"past_input_ids: {self.past_input_ids}\n")

    shard_hidden_states, shard_past_kvs, shard_logits = await self.async_forward(
      input_ids=self.past_input_ids,
      attention_mask=input_attention_mask
    )

    if DEBUG >= 4:
      print(f"\nshard_hidden_states: {shard_hidden_states}\n")
      print(f"\nshard_past_kvs {shard_past_kvs}\n")
      print(f"\nshard_logits: {shard_logits}")

    next_token = None
    if shard_logits is not None:
      next_token = await self.async_logit_sample(shard_logits)
      self.past_input_ids = torch.cat([input_ids, next_token[:, None].squeeze(-1)], dim=-1)
      input_ids = next_token

      if DEBUG >= 4:
        print(f"\nnext_token: {next_token}")

    if self.past_input_ids is not None:
      cached_iids = {"input_ids": self.past_input_ids.tolist()}

    is_finished = False
    if next_token is not None:
      is_finished = next_token.item() == self.tokenizer.eos_token_id

    return_values = (
      input_ids.numpy(force=True) if shard_logits is not None else shard_hidden_states.numpy(force=True),
      json.dumps({"cached_iids": cached_iids}),
      is_finished
    )

    if DEBUG >= 4:
      print(f"return_values: {return_values}")

    return return_values

  async def infer_tensor(
   self,
   request_id: str,
   shard: Shard,
   input_data: np.ndarray,
   inference_state: Optional[str] = None
  ) -> Tuple[np.ndarray, str, bool]:
    """
    Asynchronously processes input tensor data using the specified shard and returns the inference result.

    Args:
        request_id (str): The unique identifier for the request.
        shard (Shard): The model shard used for inference.
        input_data (np.ndarray): The input data in NumPy array format to be processed by the model.
        inference_state (str, optional): The cached inference state for resuming or continuing inference. Defaults to None.

    Returns:
        A tuple containing:

        - input_ids (np.ndarray): The processed token IDs as a NumPy array if logits were generated. Otherwise, it returns hidden states.
        - cache_json (str): A JSON string containing the cached input IDs for further inference steps.
        - is_finished (bool): A boolean indicating whether the model has reached the end-of-sequence (EOS) token.
    """
    if DEBUG >= 4:
      print("infer_tensor called")
      print(f"input_data: {input_data}")
      print(f"shard: {shard}")
      print(f"inference_state: {inference_state}")

    await self.ensure_shard(shard)

    input_ids = torch.tensor(input_data).to(self.device)

    # get cache from inference_state
    past_iids, cached_iids = self.infer_caching(inference_state)

    # detect if hidden_states or not
    hidden_states = None
    self.past_input_ids = None
    if input_ids.size()[-1] > 1:
      hidden_states = input_ids
      self.past_input_ids = past_iids
    else:
      if past_iids is not None:
        self.past_input_ids = past_iids
      else:
        self.past_input_ids = input_ids

    if DEBUG >= 4:
      print(f"\npast_input_ids: {self.past_input_ids}")
      print(f"\nhidden_state: {hidden_states}")
      print(f"\ninference_state: {inference_state}")

    shard_hidden_states, shard_past_kvs, shard_logits = await self.async_forward(
      input_ids=self.past_input_ids,
      hidden_states=hidden_states
    )

    next_token = None
    if shard_logits is not None:
      next_token = await self.async_logit_sample(shard_logits)
      input_ids = next_token

    #cache
    next_cached_logits = None
    if next_token is not None:
      if self.past_input_ids is not None:
        next_cached_logits = torch.cat([self.past_input_ids, next_token], dim=-1).to(self.device)
      elif past_iids is not None:
        next_cached_logits = torch.cat([past_iids, next_token], dim=-1).to(self.device)

      cached_iids = {
        "input_ids": next_cached_logits.tolist() if next_cached_logits is not None else []
      }

    is_finished = False
    if next_token is not None:
      is_finished = next_token.item() == self.tokenizer.eos_token_id

    if is_finished:
      # clear cache
      cached_iids = {"input_ids": []}

    if DEBUG >= 4:
      print(f"\ninput_ids: {input_ids}")
      print(f"\nshard_hidden_states: {shard_hidden_states}\n")
      print(f"\nshard_past_kvs {shard_past_kvs}\n")
      print(f"\nshard_logits: {shard_logits}")

    return_values = (
      input_ids.numpy(force=True) if shard_logits is not None else shard_hidden_states.numpy(force=True),
      json.dumps({"cached_iids": cached_iids}),
      is_finished
    )

    if DEBUG >= 4:
      print(f"return_values: {return_values}")

    return return_values

  async def ensure_shard(self, shard: Shard):
    """
    Ensure the model shard is loaded and ready for inference.

    Args:
      shard (Optional[Shard]): Shard information for the model.
    """
    if self.shard == shard:
      return

    if DEBUG >= 4:
      print(f"Loading new shard: {shard}")

    model_path = await self.shard_downloader.ensure_shard(shard)

    # get model weight map
    model_wm = await get_weight_map(repo_id=shard.model_id)

    self.stateful_sharded_model = ShardedHuggingFaceModel(
      shard=shard,
      local_model_path=model_path,
      weight_map=model_wm,
      device=self.device,
      dtype=self.dtype,
      device_map=self.device_map,
      top_k=TOP_K,
      temp=TEMP,
      top_p=TOP_P
    )
    self.shard = shard

    self.tokenizer = await resolve_tokenizer(shard.model_id)

    if DEBUG >= 4:
      print(f"Shard loaded successfully: {shard}")
