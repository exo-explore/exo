import torch
import torch.nn as nn
import numpy as np
import re

from transformers import AutoModelForCausalLM, BitsAndBytesConfig, DynamicCache, Cache
from exo.inference.shard import Shard
from exo.helpers import DEBUG
from typing import Tuple, Optional, Union, List

from exo.inference.pytorch.model.utils import sample_logits

TOP_P = 0.75 #0.95
TOP_K = 20
TEMP = 0.8

class ShardedHuggingFaceModel(torch.nn.Module):
    def __init__(self, shard: Shard, tokenizer: any):
        super(ShardedHuggingFaceModel, self).__init__()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else: 
            self.device = torch.device("cpu")

        self.shard = shard
        self.tokenizer = tokenizer

        # Load the model
        try:
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                shard.model_id,
                torch_dtype=torch.float32,
                device_map="auto",
                # offload_buffers=True
            )

            self.base_model = self.llm_model.model
        except Exception as err:
            print(f"Error loading model: {err}")
            raise

        if DEBUG >= 2:
            print(f"\nShardedHuggingFaceModel init with shard {shard}")
            print(f"self.llm_model: {self.llm_model}")
            print(f"self.llm_model.model: {self.llm_model.model}")

        # load layers from base model to use
        layers = []
        for i in range(shard.start_layer, shard.end_layer + 1):
            layer = self.llm_model.model.layers[i]

            if DEBUG >= 2:
                print(f"Loading layers[{i}]")

            layers.append(layer)
        
        self.layers = nn.ModuleList(layers)

        if DEBUG >= 2:
            print(f"full_model.model layer: {len(self.llm_model.model.layers)}")

        # Embeddings and final layer norm
        # used for doing what forward LlamaModel does in transformers
        self.norm = self.llm_model.model.norm
        self.lm_head = self.llm_model.lm_head
        self.embed_tokens = self.base_model.embed_tokens
    
    def forward(
        self,
        input_ids: torch.tensor,
        past_kvs: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    ) -> Tuple[np.ndarray, any]:
        """
        Forward through layers using the base model

        Args:
            input_ids: tensor input
            past_kvs: past key value stores for cache
            use_cache: use cache
        
        Returns:
            hidden_states: numpy of states between layers
            or logits: numpy of normalization and linearization of last hidden state
            past_kvs: DynamicCache of past key values if use_cache is true

        Ref:
            https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/qwen2/modeling_qwen2.py#L804
            https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/llama/modeling_llama.py#L887
        """
        if DEBUG >= 4:
            print("forward called")
            print(f"input_ids: {input_ids}\n")
            print(f"layer_count: {self.shard.get_layer_count()}")
            print(f"is_first_layer: {self.shard.is_first_layer()}")
            print(f"is_last_layer: {self.shard.is_last_layer()}")

        past_kvs = DynamicCache.from_legacy_cache(past_kvs)
        past_seen_tokens = past_kvs.get_seq_length() if past_kvs is not None else 0

        cache_position = torch.arange(
            past_seen_tokens, 
            past_seen_tokens + input_ids.shape[1], 
            device=input_ids.device
        ).to(self.device)

        position_ids = cache_position.unsqueeze(0).to(self.device)

        # progress through layers
        for decoder_layer in self.layers:
            if DEBUG >= 4:
                print("Going through layer")
                print(f"{decoder_layer}")

            layer_outputs = decoder_layer(
                input_ids,
                position_ids=position_ids,
                past_key_value=past_kvs,
                use_cache=True,
                cache_position=cache_position,
            )

        hidden_states = layer_outputs[0]
        next_kvs = layer_outputs[1]

        if DEBUG >= 3:
            print(f"hidden_state: {hidden_states}")
            print(f"next_kvs: {next_kvs}")
        
        if self.shard.is_last_layer():
            hs_norm = self.norm(hidden_states)
            hs_lm_head = self.llm_model.lm_head(hs_norm).float()

            # Use the sampling function with default settings
            with torch.no_grad():
                output_token = sample_logits(
                    hs_lm_head[:, -1, :],
                    TEMP,
                    TOP_P,
                    TOP_K
                ).cpu().numpy().flatten()

            if DEBUG >= 2:
                print(f"hs_norm: {hs_norm}")
                print(f"hs_lm_head: {hs_lm_head}")
                print(f"output_token: {output_token}")

            return (output_token, next_kvs)
        
        with torch.no_grad():
            out_hidden_states = hidden_states.cpu().numpy()

        return (
            out_hidden_states,
            next_kvs
        )

    # def forward_layers(
    #     self,
    #     input_data: torch.tensor
    # ) -> np.ndarray:
    #     """
    #     Forward pass through the specified layers.
    #     This is without caching

    #     Note: past_key_values not working for model, might be a library bug
    #     """ 
    #     if DEBUG >= 2:
    #         print("forward_layer call")
    #         print(f"input_data: {input_data}")
    #         print(f"shard {self.shard.to_dict()}")

    #     hidden_states = input_data

    #     # Forward pass through the layer
    #     if DEBUG >= 2:
    #         print(f"\n[layer model] {self.llm_model.model}")
    #         print(f"IN hidden_states {hidden_states}")
        
    #     layer_outputs = self.llm_model.model(
    #         hidden_states.to(self.device),
    #         use_cache=False
    #     )

    #     if DEBUG >= 2:
    #         print(f"OUT hidden_states {layer_outputs.last_hidden_state}")
        
    #     hidden_states = layer_outputs.last_hidden_state

    #     print(f"2 is_last_layer {self.shard.is_last_layer()}")
    #     if self.shard.is_last_layer():
    #         hs_norm = self.norm(hidden_states)
    #         hs_lm_head = self.llm_model.lm_head(hs_norm).float()

    #         # Use the sampling function with default settings
    #         with torch.no_grad():
    #             output_token = sample_logits(
    #                 hs_lm_head[:, -1, :],
    #                 TEMP,
    #                 TOP_P,
    #                 TOP_K
    #             ).cpu().numpy().flatten()

    #         if DEBUG >= 2:
    #             print(f"hs_norm: {hs_norm}")
    #             print(f"hs_lm_head: {hs_lm_head}")
    #             print(f"output_token: {output_token}")

    #         return output_token
        
    #     return hidden_states.cpu().numpy()
    
    # def forward_layers_cached(
    #     self,
    #     input_data: torch.tensor,
    #     past_kvs
    # ) -> Tuple[np.ndarray, list]:
    #     """
    #     Forward pass through the specified layers.
    #     With caching

    #     Note: past_key_values not working for model, might be a library bug
    #     """ 

    #     if not past_kvs:
    #         past_kvs = DynamicCache()
    #     else:
    #         past_kvs = DynamicCache.from_legacy_cache(past_kvs)
            
    #     if DEBUG >= 2:
    #         print("forward_layer call")
    #         print(f"input_data: {input_data}")
    #         print(f"shard {self.shard.to_dict()}")
    #         print(f"past_kvs: {past_kvs}")

    #     input_ids = input_data.to(self.device)
    #     position_ids = None
    #     # position_embeddings = None

    #     inputs_embeds = self.embed_tokens(input_ids)

    #     if self.shard.is_first_layer():
    #         hidden_states = self.embed_tokens(hidden_states)

    #         if DEBUG >= 2:
    #             print(f"hidden_states: {hidden_states}")
    #             print(f"hidden_states.size(): {hidden_states.size()}")

    #         batch_size, seq_len = input_data.size()
    #         position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0).expand(batch_size, -1)

    #         # check if model does not have rotary emb
    #         # have to apply rotary per model
    #         # embedding seems very model specific and using position_ids
    #         # seems more universal, even though some give warning about it
    #         # if re.match(r"Qwen|qwen", self.shard.model_id):
    #         #     import transformers.models.qwen2.modeling_qwen2 as qwen2
    #         #     position_embeddings =
    #         #         q=hidden_states,
    #         #         position_ids=position_ids
    #         #     )
    #         # else:
    #         #     position_embeddings = self.llm_model.model.rotary_emb(
    #         #         hidden_states,
    #         #         position_ids
    #         #     )

    #         # if DEBUG >= 2:
    #         #     print(f"embedded hidden_states {hidden_states}")
    #         #     print(f"position_ids: {position_embeddings}")

        
    #     # Forward pass through the layer
    #     if DEBUG >= 2:
    #         print(f"IN hidden_states {hidden_states}")
    #         print(f"past_kvs {past_kvs}")
        
    #     layer_outputs = self.llm_model.model(
    #         hidden_states,
    #         position_ids=position_ids,
    #         past_key_values=past_kvs,
    #         use_cache=True
    #     )

    #     if DEBUG >= 2:
    #         print(f"\nlayer_outputs: {layer_outputs}")
        
    #     hidden_states = layer_outputs.last_hidden_state
    #     present_kvs = layer_outputs.past_key_values

    #     print(f"2 is_last_layer {self.shard.is_last_layer()}")
    #     if self.shard.is_last_layer():
    #         hs_norm = self.norm(hidden_states)
    #         hs_lm_head = self.llm_model.lm_head(hs_norm).float()

    #         # Use the sampling function with default settings
    #         with torch.no_grad():
    #             output_token = sample_logits(
    #                 hs_lm_head[:, -1, :],
    #                 TEMP,
    #                 TOP_P,
    #                 TOP_K
    #             ).cpu().numpy().flatten()

    #         if DEBUG >= 2:
    #             print(f"hs_norm: {hs_norm}")
    #             print(f"hs_lm_head: {hs_lm_head}")
    #             print(f"output_token: {output_token}")

    #         return (output_token, present_kvs)
        
    #     return (hidden_states.cpu().numpy(), present_kvs)