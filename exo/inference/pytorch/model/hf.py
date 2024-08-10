import torch
from transformers import AutoModelForCausalLM
from exo.inference.shard import Shard
from exo.helpers import DEBUG
from typing import Tuple

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
        input_data: torch.tensor
    ) -> any:
        """
        Forward pass through the specified layers.

        Note: past_key_values not working for model, might be a library bug
        """ 
        if DEBUG >= 2:
            print("forward_layer call")
            print(f"input_data: {input_data}")
            print(f"1 shard {self.shard.to_dict()}")

        # Initialize position_ids
        position_ids = torch.arange(
            input_data.size(1),
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)

        hidden_states = input_data

        if self.shard.is_first_layer():
            hidden_states = self.embed_tokens(hidden_states)
            if DEBUG >= 2:
                print(f"embedded hidden_states {hidden_states}")

        for i, layer in enumerate(self.layers):
            # Forward pass through the layer
            if DEBUG >= 2:
                print(f"\n[layer {i}] {layer}")
                print(f"hidden_states {hidden_states}")
            
            layer_outputs = layer(
                hidden_states,
                position_ids=position_ids
            )

            if DEBUG >= 2:
                print(f"\n[layer {i}] layer_outputs: {layer_outputs[0]}")
            
            hidden_states = layer_outputs[0]

        print(f"2 is_last_layer {self.shard.is_last_layer()}")
        if self.shard.is_last_layer():
            hs_norm = self.norm(hidden_states)
            hs_lm_head = self.full_model.lm_head(hs_norm).float().flatten()
            
            if DEBUG >= 2:
                print(f"hs_norm: {hs_norm}")
                print(f"hs_lm_head: {hs_lm_head}")

            return (hs_lm_head, hidden_states)
        
        return hidden_states
