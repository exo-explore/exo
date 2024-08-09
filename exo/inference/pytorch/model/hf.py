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

    # def prefill(self, tokens: list[int], start_pos: int=0) -> int:
    #     print(f"\nprefill called")
    #     """
    #     Process the initial input tokens and set up the initial hidden states.
    #     """
    #     # Assuming tokens is a 1D tensor of token IDs
    #     for token in tokens:
    #         # Convert token to a tensor and get embeddings
    #         token_tensor = torch.tensor([[token]], device=self.device)
    #         token_tensor = self.embed_tokens(token_tensor)
                
    #         if DEBUG >= 2:
    #             print(f"\ntoken_tensor shape: {token_tensor.shape}")

    #         # Prefill with tokens
    #         self.forward_layers(start_pos, token_tensor, None)

    #         # Increment start position
    #         start_pos += 1

    #     return start_pos

    def forward_layers(
        self,
        input_data: torch.tensor,
        #past_key_values: list
    ) -> torch.tensor: #-> Tuple[torch.tensor, list]:
        """
        Forward pass through the specified layers.

        Note: past_key_values not working for model, might be a library bug
        """ 
        if DEBUG >= 2:
            print("forward_layer call")
            print(f"input_data: {input_data}")
            print(f"1 shard {self.shard.to_dict()}")

        # Check past key values
        # if past_key_values is None:
        #     past_key_values = [None] * len(self.layers)

        # Initialize position_ids
        position_ids = torch.arange(
            input_data.size(1),
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)

        #new_past_key_values = []
        hidden_states = input_data
        for i, layer in enumerate(self.layers):
            # Forward pass through the layer
            if DEBUG >= 2:
                print(f"\n[layer {i}] {layer}")
                print(f"hidden_states {hidden_states}")

            # Get past key value if available
            # past_key_value = past_key_values[i] if past_key_values and len(past_key_values) > 0 else None

            # embed only at first layer
            if i == 0:
                hidden_states = self.embed_tokens(hidden_states)
                if DEBUG >= 2:
                    print(f"embedded hidden_states {hidden_states}")
            
            layer_outputs = layer(
                hidden_states,
                position_ids=position_ids,
                # past_key_value=past_key_value,
                # use_cache=True
            )

            if DEBUG >= 2:
                print(f"\n[layer {i}] layer_outputs: {layer_outputs[0]}")
            
            hidden_states = layer_outputs[0]

            if DEBUG >= 2:
                print(f"2 is last layer? {self.shard.is_last_layer()}")
                print(f"2 shard {self.shard.to_dict()}")

            if self.shard.is_last_layer():
                # output_data = output_data.view(1, -1, 4096)
                return self.norm(hidden_states)

        return hidden_states
        # if self.shard.is_last_layer():
        #     logits = self.full_model.model.norm(hidden_states)
        #     return logits.flatten() #, new_past_key_values
        # else:
        #     return hidden_states#, new_past_key_values

