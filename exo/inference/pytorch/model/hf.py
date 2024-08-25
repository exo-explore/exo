import torch
import torch.nn as nn
import numpy as np

from transformers import AutoModelForCausalLM, BitsAndBytesConfig, DynamicCache, Cache
from exo.inference.shard import Shard
from exo.helpers import DEBUG
from typing import Tuple

from .utils import sample_logits

TOP_P = 0.75 #0.95
TOP_K = 20
TEMP = 0.8

class ShardedHuggingFaceModel(torch.nn.Module):
    def __init__(self, shard: Shard):
        super(ShardedHuggingFaceModel, self).__init__()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else: 
            self.device = torch.device("cpu")

        self.shard = shard

        # Load the model
        try:
            self.full_model = AutoModelForCausalLM.from_pretrained(
                shard.model_id,
                torch_dtype="auto",
                device_map="auto",
                # offload_buffers=True
            )
            # .to(self.device)
        except Exception as err:
            print(f"Error loading model: {err}")
            raise

        if DEBUG >= 2:
            print(f"\nShardedHuggingFaceModel init with shard {shard}")
            print(f"self.full_model: {self.full_model}")
            print(f"self.full_model.model: {self.full_model.model}")

        # using llamaconfig not working setting layers manually
        layers = []
        for i in range(shard.start_layer, shard.end_layer + 1):
            layer = self.full_model.model.layers[i]

            if DEBUG >= 2:
                print(f"Loading layers[{i}]")

            layers.append(layer)
        
        self.full_model.model.layers = nn.ModuleList(layers)
        # .to(self.device)

        if DEBUG >= 2:
            print(f"full_model.model layer: {len(self.full_model.model.layers)}")

        # Embeddings and final layer norm
        # used for doing what forward LlamaModel does in transformers
        self.embed_tokens = self.full_model.model.embed_tokens
        self.norm = self.full_model.model.norm

    def forward_layers(
        self,
        input_data: torch.tensor
    ) -> np.ndarray:
        """
        Forward pass through the specified layers.
        This is without caching

        Note: past_key_values not working for model, might be a library bug
        """ 
        if DEBUG >= 2:
            print("forward_layer call")
            print(f"input_data: {input_data}")
            print(f"shard {self.shard.to_dict()}")

        hidden_states = input_data

        # Forward pass through the layer
        if DEBUG >= 2:
            print(f"\n[layer model] {self.full_model.model}")
            print(f"IN hidden_states {hidden_states}")
        
        layer_outputs = self.full_model.model(
            hidden_states.to(self.device),
            use_cache=False
        )

        if DEBUG >= 2:
            print(f"OUT hidden_states {layer_outputs.last_hidden_state}")
        
        hidden_states = layer_outputs.last_hidden_state

        print(f"2 is_last_layer {self.shard.is_last_layer()}")
        if self.shard.is_last_layer():
            hs_norm = self.norm(hidden_states)
            hs_lm_head = self.full_model.lm_head(hs_norm).float()

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

            return output_token
        
        return hidden_states.cpu().numpy()
    
    def forward_layers_cached(
        self,
        input_data: torch.tensor,
        past_kvs: Cache = DynamicCache()
    ) -> Tuple[np.ndarray, list]:
        """
        Forward pass through the specified layers.
        With caching

        Note: past_key_values not working for model, might be a library bug
        """ 
        if DEBUG >= 2:
            print("forward_layer call")
            print(f"input_data: {input_data}")
            print(f"shard {self.shard.to_dict()}")

        hidden_states = input_data
        position_ids = None
        position_embeddings = None

        if self.shard.is_first_layer():
            hidden_states = self.embed_tokens(hidden_states)

            if DEBUG >= 2:
                print(f"hidden_states: {hidden_states}")
                print(f"hidden_states.size(): {hidden_states.size()}")

            batch_size, seq_len = input_data.size()
            position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0).expand(batch_size, -1)

            position_embeddings = self.full_model.model.rotary_emb(
                hidden_states,
                position_ids
            )

            # if DEBUG >= 2:
            #     print(f"embedded hidden_states {hidden_states}")
            #     print(f"position_ids: {position_embeddings}")

        
        # Forward pass through the layer
        if DEBUG >= 2:
            print(f"IN hidden_states {hidden_states}")
            print(f"past_kvs {past_kvs}")
        
        layer_outputs = self.full_model.model(
            hidden_states,
            position_ids=position_ids,
            inputs_embeds=position_embeddings,
            past_key_values=past_kvs,
            use_cache=True
        )

        if DEBUG >= 2:
            print(f"\nlayer_outputs: {layer_outputs}")
        
        hidden_states = layer_outputs.last_hidden_state
        present_kvs = layer_outputs.past_key_values

        print(f"2 is_last_layer {self.shard.is_last_layer()}")
        if self.shard.is_last_layer():
            hs_norm = self.norm(hidden_states)
            hs_lm_head = self.full_model.lm_head(hs_norm).float()

            # Use the sampling function with default settings
            output_token = sample_logits(
                hs_lm_head[:, -1, :],
                TEMP,
                TOP_P,
                TOP_K
            ).numpy()

            if DEBUG >= 2:
                print(f"hs_norm: {hs_norm}")
                print(f"hs_lm_head: {hs_lm_head}")
                print(f"output_token: {output_token}")

            return (output_token, present_kvs)
        
        return (hidden_states.numpy(), present_kvs)