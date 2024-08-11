import torch
import numpy as np

from transformers import AutoModelForCausalLM, LlamaConfig, DynamicCache, Cache
from exo.inference.shard import Shard
from exo.helpers import DEBUG
from typing import Tuple

from .utils import sample_logits

class ShardedHuggingFaceModel(torch.nn.Module):
    def __init__(self, shard: Shard):
        super(ShardedHuggingFaceModel, self).__init__()

        if DEBUG >= 2:
            print(f"\nShardedHuggingFaceModel init with shard {shard}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.shard = shard

        

        # Load the model
        self.full_model = AutoModelForCausalLM.from_pretrained(
            shard.model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

        # set model config to restrict layers and enable caching
        self.config = LlamaConfig.from_pretrained(shard.model_id)
        # self.config.use_cache = True  # Enable caching

        # Extract only the layers for this shard
        # get layers up to end layer
        self.config.num_hidden_layers = 2
        self.full_model.model.config = self.config
        if DEBUG >= 2:
            print(f"full_model.model layer: {len(self.full_model.model.layers)}")

        # Embeddings and final layer norm
        # used for doing what forward LlamaModel does in transformers
        self.embed_tokens = self.full_model.model.embed_tokens
        self.norm = self.full_model.model.norm

        # self.past_key_values = DynamicCache()

    def forward_layers(
        self,
        input_data: torch.tensor
    ) -> Tuple[np.ndarray, list]:
        """
        Forward pass through the specified layers.

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
            print(f"hidden_states {hidden_states}")
            # print(f"past_kvs {past_kvs}")
        
        layer_outputs = self.full_model.model(
            hidden_states,
            # position_ids=position_ids,
            # inputs_embeds=position_embeddings,
            # past_key_values=self.past_key_values,
            # use_cache=True # not enough vram for using cache ;_;
        )

        if DEBUG >= 2:
            print(f"\nlayer_outputs: {layer_outputs}")
        
        hidden_states = layer_outputs.last_hidden_state
        # self.past_key_values = layer_outputs.past_key_values

        print(f"2 is_last_layer {self.shard.is_last_layer()}")
        if self.shard.is_last_layer():
            hs_norm = self.norm(hidden_states)
            hs_lm_head = self.full_model.lm_head(hs_norm).float()

            # Use the sampling function with default settings
            output_token = sample_logits(
                hs_lm_head[:, -1, :]).cpu().numpy().flatten()

            if DEBUG >= 2:
                print(f"hs_norm: {hs_norm}")
                print(f"hs_lm_head: {hs_lm_head}")
                print(f"output_token: {output_token}")

            return output_token
        
        return hidden_states.cpu().numpy()
