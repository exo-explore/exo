import os
import json
from typing import Tuple, Optional, Union, List
from pathlib import Path

import torch
import torch.nn as nn

from exo.inference.shard import Shard
from exo.helpers import DEBUG
from exo.inference.torch.utils import extract_layers

from transformers import (
  AutoConfig,
  AutoModelForCausalLM,
  DynamicCache,
  Cache,
  LogitsProcessorList,
  TopKLogitsWarper,
  TopPLogitsWarper,
  TemperatureLogitsWarper
)

# llama
from transformers.models.llama.modeling_llama import LlamaModel

class ShardedHuggingFaceModel:
  def __init__(
    self,
    shard: Shard,
    local_model_path: Path,
    weight_map: Optional[dict],
    device: torch.device,
    dtype: torch.dtype,
    device_map: str,
    top_k: int = 25,
    temp: float = 0.7,
    top_p: float = 0.9,
    offload_buffers: bool = True
  ):
    """
    Initializes the ShardedHuggingFaceModel with a specified shard, model path, and device.

    Args:
        shard (Shard): The model shard containing the start and end layers.
        local_model_path (str): The local path to the model.
        device (str): The device on which to run the model, e.g., "cuda" or "cpu".
        dtype (torch.dtype): The data type (precision) to be used for model computations.
        top_k (int, optional): The number of top tokens to consider for sampling. Defaults to 25.
        temp (float, optional): The temperature for softmax sampling. Defaults to 0.7.
        top_p (float, optional): The cumulative probability threshold for nucleus sampling. Defaults to 0.9.
    """

    # class vars
    self.shard = shard
    self.hidden_states = None
    self.input_ids = None
    self.inputs_embeds = None
    self.attention_mask = None
    self.position_embeddings = None
    self.past_key_values = None
    self.cache_position = None
    self.position_ids = None
    self.causal_mask = None
    self.local_model_path = local_model_path

    # setup logit processors
    self.logits_processor = LogitsProcessorList([
      TopKLogitsWarper(top_k),
      TemperatureLogitsWarper(temp),
      TopPLogitsWarper(top_p)
    ])

    self.device = device
    self.dtype = dtype
    self.device_map = device_map

    self.offload_buffers = offload_buffers

    self.model_safetensors_path = self.local_model_path/"model.safetensors.index.json"

    # setup pytorch and transformer llm
    try:
      if weight_map:
        print("loading shard model")
        self.llm_model = self.load_sharded_model(
          shard,
          weight_map,
          offload_buffers=self.offload_buffers
        )

        # clear out edited safetensor json
        # this is needed because shard downloader just
        # appends and not redownloads the file
        os.remove(self.model_safetensors_path)
      else:
        print("loading full model")
        self.llm_model = AutoModelForCausalLM.from_pretrained(
          pretrained_model_name_or_path=self.local_model_path,
          torch_dtype=self.dtype,
          device_map=self.device_map,
          offload_buffers=True
        ).to(self.device)

      self.model = self.llm_model.model.to(self.device)
    except Exception as err:
      print(f"error loading and splitting model: {err}")
      raise

  def load_sharded_model(
    self,
    shard: Shard,
    weight_map: dict,
    offload_buffers: bool
  ) -> AutoModelForCausalLM:
    """
    Loads sharded version of model where only needed
    weights are loaded for necessary layers

    Args:

    Returns:
    """
    if DEBUG >= 4:
      print("load_sharded_model called")
      print(f"shard: {shard}")

    # break out layers per shard range
    layer_weight_map = extract_layers(
      weight_map,
      shard
    )

    # rewrite model.safetensors.index.json for only needed layers
    try:
      mst_json = {}
      with open(self.model_safetensors_path, "r") as mst_file:
        mst_json = json.load(mst_file)
        mst_json["weight_map"] = layer_weight_map

      if DEBUG >= 4:
        print(f"rewritten safetensor index \n{json.dumps(mst_json, indent=4)}")

      os.remove(self.model_safetensors_path)

      with open(self.model_safetensors_path, "w") as mst_file:
        json.dump(mst_json, mst_file, indent=4)
    except Exception as err:
      print(f"err: {err}")
      raise

    # load model
    try:
      shard_num_hidden_layers = shard.end_layer - shard.start_layer
      if DEBUG >= 4:
        print(f"config with {shard_num_hidden_layers} layers")
      return AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=self.local_model_path,
        device_map=self.device_map,
        torch_dtype=self.dtype,
        offload_buffers=offload_buffers,
        local_files_only=True,
        num_hidden_layers=shard_num_hidden_layers
      ).to(self.device)
    except Exception as err:
      print(f"err: {err}")
      raise

  def forward(
    self,
    input_ids: Optional[torch.Tensor] = None,
    hidden_states: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    use_legacy_cache: bool = False
  ) -> Tuple[Optional[torch.Tensor], Optional[Union[Cache, List[torch.FloatTensor]]], Optional[torch.Tensor]]:
    """
    Performs a forward pass through the model shard, computing hidden states, past key values, and logits.

    Args:
        input_ids (torch.Tensor, optional): The input token IDs for the model. Either input_ids or hidden_states must be provided.
        hidden_states (torch.Tensor, optional): The hidden states of the model at the current layer.
        attention_mask (torch.Tensor, optional): The attention mask to prevent attending to padding tokens.
        past_key_values (Union[Cache, List[torch.FloatTensor]], optional): Cached past key values for fast autoregressive generation.
        use_legacy_cache (bool, optional): Whether to use the legacy cache format for past key values. Defaults to False.

    Returns:
        Tuple:
            - hidden_states (torch.Tensor, optional): The hidden states after the forward pass.
            - past_key_values (Union[Cache, List[torch.FloatTensor]], optional): The updated past key values.
            - logits (torch.Tensor, optional): The logits produced by the model if the last layer is processed.
    """
    model_inputs = None
    self.hidden_states = hidden_states
    self.input_ids = input_ids

    # if there is hidden states and no position_ids, will need to be calculated
    # this is not needed for Qwen model but Llama requires it

    # embed input_ids
    self.inputs_embeds = self.model.embed_tokens(self.input_ids)

    # cache
    if past_key_values and not isinstance(past_key_values, Cache):
      use_legacy_cache = True
      past_key_values = DynamicCache.from_legacy_cache(past_key_values)

    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    cache_position = torch.arange(
      past_seen_tokens,
      past_seen_tokens + self.inputs_embeds.shape[1],
      device=self.inputs_embeds.device
    )

    # position id
    self.position_ids = cache_position.unsqueeze(0)

    if DEBUG >= 4:
      print("hf forward called")
      print(f"hidden_states: {self.hidden_states}")
      print(f"input_ids: {self.input_ids}")
      print(f"input_embeds: {self.inputs_embeds}")
      print(f"position_ids: {self.position_ids}")
      print(f"past_key_values: {past_key_values}")

    if self.hidden_states is None:
      # casual mask and attention_mask
      self.attention_mask = attention_mask
      self.causal_mask = self.model._update_causal_mask(
        None,
        self.inputs_embeds,
        cache_position,
        past_key_values,
        False # dont out attentions
      )

      # embed positions, some models require and some dont
      if isinstance(self.model, LlamaModel):
        self.position_embeddings = self.model.rotary_emb(
          self.inputs_embeds,
          self.position_ids
        )

      # prepare inputs for decoder layers
      model_inputs = self.llm_model.prepare_inputs_for_generation(
        self.input_ids,
        past_key_values=past_key_values,
        attention_mask=self.attention_mask,
        inputs_embeds=self.inputs_embeds,
        position_ids=self.position_ids,
        cache_position=cache_position
      )

      self.hidden_states = self.inputs_embeds
      self.position_ids = model_inputs["position_ids"]
      self.cache_position = model_inputs["cache_position"]
      self.past_key_values = model_inputs["past_key_values"]

      if DEBUG >= 4:
        print(f"model_inputs: {model_inputs}")

    # run through decoder layers
    layer_amt = range(self.shard.end_layer - self.shard.start_layer)

    if DEBUG >= 4:
      print(f"hidden_states: {self.hidden_states}")
      print(f"model layer amt: {len(self.model.layers)}")
      print(f"layer_amt: {layer_amt}")

    for i in layer_amt:
      decoder_layer = self.model.layers[i]
      if DEBUG >= 5:
        print(f"layer #{i}")
        print("decoder_layer before")
        print(f"decoder_layer: {decoder_layer}")
        print(f"hidden_states: {self.hidden_states}")
        print(f"position_ids: {self.position_ids}")
        print(f"position_embeddings: {self.position_embeddings}")

      # TODO: fix caching as decoder layer is not returning
      # present_key_value from attention layer on models
      # might have some other generation functions needed to do it
      # see https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L2917
      # for qwen2 exhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py#L291
      layer_outputs = decoder_layer(
        self.hidden_states,
        attention_mask=self.causal_mask,
        position_ids=self.position_ids,
        past_key_values=self.past_key_values,
        use_cache=True,
        cache_position=self.cache_position
      )

      self.hidden_states = layer_outputs[0]
      self.next_decoder_cache = layer_outputs[1]

      if DEBUG >= 5:
        print("decoder_layer after")
        print(f"layer_outputs: {layer_outputs}\n")
        print(f"self.next_decoder_cache: {self.next_decoder_cache}")
        print(f"hidden_states: {self.hidden_states}")
        print(f"next_decoder_cache: {self.next_decoder_cache}")

    # handle last layer to get logits
    # shard is last layer says true at the start and not detecting last layer correctly
    if self.shard.is_last_layer():
      self.hidden_states = self.model.norm(self.hidden_states)
      if use_legacy_cache:
        self.past_key_values = self.next_decoder_cache.to_legacy_cache()
      else:
        self.past_key_values = self.next_decoder_cache

      # lm_head
      logits = self.llm_model.lm_head(self.hidden_states).to(self.device)

      if DEBUG >= 4:
        print(f"logits: {logits}")

      return (
        None,
        None,
        logits
      )

    if DEBUG >= 4:
      print("hf out [no logit]")
      print(f"hidden_states: {self.hidden_states}")
      print(f"past_key_values: {self.past_key_values}")
      print(f"position_ids: {self.position_ids}")
      print(f"input_ids: {self.input_ids}")

    return (
      self.hidden_states,
      self.past_key_values,
      None
    )

  def logits_sample(
    self,
    logits: torch.Tensor,
    use_max: Optional[bool] = False
  ) -> torch.Tensor:
    """
    Samples the next token from the model's output logits, either by using argmax or probabilistic sampling.

    Args:
        logits (torch.Tensor): The logits output from the model's final layer.
        use_max (bool, optional): If True, uses torch.argmax to select the next token from logits. Defaults to False.

    Returns:
        torch.Tensor: The next predicted token.
    """

    # get a single cloned logit
    logits = logits[:, -1, :].clone().float()

    next_token_scores = self.logits_processor(self.input_ids, logits)

    if not use_max:
      probs = nn.functional.softmax(next_token_scores, dim=-1)
      next_token = torch.multinomial(probs, num_samples=1)
    else:
      next_token = torch.argmax(next_token_scores, dim=-1)

    if DEBUG >= 4:
      print(f"input_ids: {self.input_ids}")
      print(f"next_token: {next_token}")

    return next_token[:, None].squeeze(-1)
