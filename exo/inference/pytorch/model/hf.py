import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Union, List

from exo.inference.shard import Shard
from exo.helpers import DEBUG

from transformers import (
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
    local_model_path,
    device,
    dtype,
    top_k: int = 25,
    temp: float = 0.7,
    top_p: float = 0.9,
    max_length: int = 50,
    max_time: float = 10.0
  ):
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
    self.torch_dtype = dtype

    # setup pytorch and transformer llm
    try:
      self.llm_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=self.local_model_path,
        torch_dtype=self.torch_dtype,
        device_map="auto",
        offload_buffers=True
      )

      self.model = self.llm_model.model
    except Exception as err:
      print(f"error loading and splitting model: {err}")
      raise


  def forward(
    self,
    input_ids: Optional[torch.tensor] = None,
    hidden_states: Optional[torch.tensor] = None,
    attention_mask: Optional[torch.tensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    use_legacy_cache: bool = False
  ) -> Tuple[Optional[torch.tensor], Optional[Union[Cache, List[torch.FloatTensor]]], Optional[torch.tensor]]:
    
    """
    Generate hidden states or logits via passing through set amount of layers of a model
    To be passed only input_ids OR hidden_state and not both. This is for connecting the model
    layer to generate a complete output

    Args:
      model: base llm model tramsformers class 
      llm_model: llm chat model class 
      input_ids: tensor optional
      attention_mask: tensor optional
      past_key_values: Cache or list[tensor] optional
      use_legacy_cache: bool optional
      infer_tensor: bool optional, lets forward know to handle tensors

    Returns:
      Tuple of
        - hidden_states: tensor optional
        - past_key_values: Cache or list[tensor] optional 
        - logits: tensor Optional

    """
    model_inputs = None
    self.hidden_states = hidden_states
    self.input_ids = input_ids

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
    position_ids = cache_position.unsqueeze(0)

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
        position_ids
      )

    # prepare inputs for decoder layers
    model_inputs = self.llm_model.prepare_inputs_for_generation(
      self.input_ids,
      past_key_values=past_key_values,
      attention_mask=self.attention_mask,
      inputs_embeds=self.inputs_embeds,
      position_ids=position_ids,
      cache_position=cache_position
    )

    self.hidden_states = self.inputs_embeds
    self.position_ids = model_inputs["position_ids"]
    self.cache_position = model_inputs["cache_position"]
    self.past_key_values = model_inputs["past_key_values"]

    if DEBUG >= 4:
      print(f"model_inputs: {model_inputs}")

    # run through decoder layers 
    layer_amt = range(self.shard.start_layer, self.shard.end_layer + 1)

    if DEBUG >= 4:
      print(f"hidden_states: {self.hidden_states}")
      print(f"layer_amt: {layer_amt}")

    for i in layer_amt:
      decoder_layer = self.model.layers[i]
      if DEBUG >= 5:
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
      print(f"hidden_states: {self.hidden_states}")
      print(f"past_key_values: {self.past_key_values}")

    return (
      self.hidden_states,
      self.past_key_values,
      None
    )

  def logits_sample(
    self,
    logits: torch.tensor,
    use_max: Optional[bool] = False
  ) -> torch.tensor:
    """
    Get a sample of the logits from end of model run for next token
    
    Args:
      logits: tensor 
      use_max: bool, if function should sample with argmax

    Returns:
      next_token: tensor 
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
