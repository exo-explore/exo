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
        self.embed_tokens = self.full_model.model.embed_tokens
        self.norm = self.full_model.model.norm
        self.lm_head = self.full_model.lm_head

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
        hidden_states: torch.tensor,
        #past_key_values: list
    ) -> torch.tensor: #-> Tuple[torch.tensor, list]:
        """
        Forward pass through the specified layers.

        Note: past_key_values not working for model, might be a library bug
        """
        # Embed tensor if first layer
        # if self.shard.is_first_layer():
        #     if DEBUG >= 2:
        #         print(f"Embedding first layer input_ids {hidden_states.shape}")
            
        #     # flatten to 1d and turn to long
        #     if hidden_states.dim() > 1: 
        #         hidden_states = hidden_states.view(-1)  
        #     hidden_states = hidden_states.long()
        #     hidden_states = self.embed_tokens(hidden_states)
        # else:
        #     hidden_states = hidden_states

        # Check past key values
        # if past_key_values is None:
        #     past_key_values = [None] * len(self.layers)

        # Initialize position_ids
        position_ids = torch.arange(
            hidden_states.size(1),
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)

        #new_past_key_values = []
        for i, layer in enumerate(self.layers):
            # Get past key value if available
            # past_key_value = past_key_values[i] if past_key_values and len(past_key_values) > 0 else None
            
            # Forward pass through the layer
            if DEBUG >= 2:
                print(f"\nPass tensor to layer[{i}] {layer}")

            layer_outputs = layer(
                hidden_states,
                position_ids=position_ids,
                # past_key_value=past_key_value,
                # use_cache=True
            )

            if DEBUG >= 2:
                print(f"\nlayer_outputs: {layer_outputs}")
            
            hidden_states = layer_outputs[0]
            # new_past_key_values.append(layer_outputs[1])

        return hidden_states
        # if self.shard.is_last_layer():
        #     logits = self.full_model.model.norm(hidden_states)
        #     return logits.flatten() #, new_past_key_values
        # else:
        #     return hidden_states#, new_past_key_values

