import torch
import torch.nn as nn
import numpy as np
import gc
from typing import Tuple, Optional, Union, List

from exo.inference.shard import Shard
from exo.helpers import DEBUG
from exo.inference.inference_engine import InferenceEngine
from exo.download.shard_download import ShardDownloader

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
    Cache,
    LogitsProcessorList,
    #MinLengthLogitsProcessor,
    LogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TemperatureLogitsWarper,
    StoppingCriteriaList,
    MaxLengthCriteria,
    MaxTimeCriteria
)

from transformers.generation.configuration_utils import (
    GenerationConfig,
    GenerationMode
)

# llama 
from transformers.models.llama.modeling_llama import LlamaModel

# qwen2
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model


class ShardedHuggingFaceModel(InferenceEngine):
    def __init__(self, shard: Shard, ):
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

        # setup pytorch and transformer llm
        try:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                shard.model_id,
                torch_dtype=self.torch_dtype,
                device_map="auto"
            ) 
        except Exception as err:
            print(f"error loading and splitting model: {err}")
            raise


    def forward(
        self,
        shard: Optional[Shard] = None,
        model,
        llm_model,
        input_ids: Optional[torch.tensor] = None,
        hidden_states: Optional[torch.tensor] = None,
        attention_mask: Optional[torch.tensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        use_legacy_cache: Optional[bool] = False
    ) -> Tuple[Optional[torch.tensor], Optional[Union[Cache, List[torch.FloatTensor]]], Optional[torch.tensor]]:
        
        """
        Generate hidden states or logits via passing through set amount of layers of a model
        To be passed only input_ids OR hidden_state and not both. This is for connecting the model
        layer to generate a complete output

        Args:
            model: base llm model tramsformers class 
            llm_model: llm chat model class 
            input_ids: tensor optional
            hidden_states: tensor optional
            attention_mask: tensor optional
            past_key_values: Cache or list[tensor] optional
            use_legacy_cache: bool optional 

        Returns:
            Tuple of 
                - hidden_states: tensor optional
                - past_key_values: Cache or list[tensor] optional 
                - logits: tensor Optional

        """

        if input_ids is not None and hidden_states is not None:
            raise ValueError

        if hidden_states is not None:
            self.hidden_states = hidden_states

        if input_ids is not None:
            self.input_ids = input_ids

            # embed input_ids
            self.inputs_embeds = model.embed_tokens(self.input_ids)
        
            # cache
            if past_key_values and not isinstance(past_key_values, Cache):
                print("Using legacy cache")
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
            self.causal_mask = model._update_causal_mask(
                None,
                self.inputs_embeds,
                cache_position,
                past_key_values,
                output_attentions
            )

            # embed positions, some models require and some dont
            if isinstance(model, LlamaModel):
                self.position_embeddings = model.rotary_emb(
                    self.inputs_embeds,
                    position_ids
                )
            
            # prepare inputs for decoder layers
            model_inputs = llm_model.prepare_inputs_for_generation(
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

        # run through decoder layers 
        layer_amt = range(self.shard.start_layer, self.shard.end_layer + 1) 
        for i in layer_amt:
            decoder_layer = self.model.layers[i]
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


        # handle last layer to get logits 
        if self.is_last:
            self.hidden_states = model.norm(self.hidden_states)

            if use_legacy_cache:
                self.past_key_values = self.next_decoder_cache.to_legacy_cache()
            else:
                self.past_key_values = self.next_decoder_cache
            
            # lm_head  
            logits = llm_model.lm_head(self.hidden_states).to(self.device)

            return (
                None,
                None,
                logits
            )

        return (
            self.hidden_states,
            self.past_key_values,
            None
        )
        
