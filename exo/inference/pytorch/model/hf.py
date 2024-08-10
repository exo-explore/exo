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

        # Load the model with the configuration for caching
        self.config = LlamaConfig.from_pretrained(shard.model_id)
        self.config.use_cache = True  # Enable caching

        # Load the model
        self.full_model = AutoModelForCausalLM.from_pretrained(
            shard.model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            config=self.config
        )
        
        # Extract only the layers for this shard
        print(f"\nself.model: {self.full_model.model}\n")
        print(f"\nlayer amount: {len(self.full_model.model.layers)}")
        self.layers = []
        for i in range(shard.start_layer, shard.end_layer + 1):
            # if DEBUG >= 2:
            #     print(f"loading layer[{i}]: {self.full_model.model.layers[i]}")
            
            self.layers.append(self.full_model.model.layers[i])

        # self.layers = torch.nn.ModuleList(layer_list)

        # Embeddings and final layer norm
        # used for doing what forward LlamaModel does in transformers
        self.embed_tokens = self.full_model.model.embed_tokens
        self.norm = self.full_model.model.norm

    def forward_layers(
        self,
        input_data: torch.tensor,
        past_kvs: Cache = DynamicCache()
    ) -> Tuple[np.ndarray, list]:
        """
        Forward pass through the specified layers.

        Note: past_key_values not working for model, might be a library bug
        """ 
        if DEBUG >= 2:
            print("forward_layer call")
            print(f"input_data: {input_data}")
            print(f"1 shard {self.shard.to_dict()}")

        hidden_states = input_data
        position_ids = None
        position_embeddings = None
        present_kvs = DynamicCache()

        if self.shard.is_first_layer():
            hidden_states = self.embed_tokens(hidden_states)

            if DEBUG >= 2:
                print(f"hidden_states: {hidden_states}")
                print(f"hidden_states.size(): {hidden_states.size()}")

            # batch_size, seq_len = input_data.size()
            # position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0).expand(batch_size, -1)

            # position_embeddings = self.full_model.model.rotary_emb(
            #     hidden_states,
            #     position_ids
            # )

            # if DEBUG >= 2:
            #     print(f"embedded hidden_states {hidden_states}")
            #     print(f"position_ids: {position_embeddings}")

        for i, layer in enumerate(self.layers):
            # Forward pass through the layer
            if DEBUG >= 2:
                print(f"\n[layer {i}] {layer}")
                print(f"hidden_states {hidden_states}")
                print(f"past_kvs {past_kvs}")
            
            layer_outputs = layer(
                hidden_states,
                # position_embeddings=position_embeddings,
                past_key_values=past_kvs,
                use_cache=True
            )

            if DEBUG >= 2:
                print(f"\n[layer {i}] layer_outputs: {layer_outputs}")
            
            hidden_states = layer_outputs[0]
            present_kvs = layer_outputs[1]

            if DEBUG >= 2:
                print(f"present_kvs {present_kvs}")

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

            return (output_token, present_kvs)
        
        return (hidden_states.cpu().numpy(), present_kvs)
