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
        self.layers = torch.nn.ModuleList([
            self.full_model.model.layers[i] for i in range(shard.start_layer, shard.end_layer + 1)
        ])

        # Embeddings and final layer norm
        self.embed_tokens = self.full_model.model.embed_tokens
        self.norm = self.full_model.model.norm
        self.lm_head = self.full_model.lm_head

    def prefill(self, tokens: list[int], start_pos: int=0) -> int:
        print(f"\nprefill called")
        """
        Process the initial input tokens and set up the initial hidden states.
        """
        # Assuming tokens is a 1D tensor of token IDs
        for token in tokens:
            # Convert token to a tensor and get embeddings
            token_tensor = torch.tensor([[token]], device=self.device)
            token_tensor = self.embed_tokens(token_tensor)
                
            if DEBUG >= 2:
                print(f"\ntoken_tensor shape: {token_tensor.shape}")

            # Prefill with tokens
            self.forward_layers(start_pos, token_tensor, None)

            # Increment start position
            start_pos += 1

        return start_pos

    def forward_layers(
            self,
            start_pos: int,
            in_tensor: torch.tensor,
            past_key_values=None
        ) -> Tuple[any, list]:

        """
        Forward pass through the specified layers.
        """
        # embed in_tensor
        # in_tensor = self.embed_tokens(in_tensor)

        # check past key values
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        # Initialize position_ids
        position_ids = torch.arange(
            start_pos, 
            start_pos + len(past_key_values), 
            dtype=torch.long, 
            device=self.device
        )
        position_ids = position_ids.unsqueeze(0)

        new_past_key_values = []
        out_tensor = None
        for i, layer in enumerate(self.layers):
            # Get past key value if available
            if past_key_values and len(past_key_values) > 0:
                past_key_value = past_key_values[i] 
            else:
                past_key_value = None
            
            # Forward pass through the layer
            layer_outputs = layer(
                in_tensor if not out_tensor else out_tensor,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=True,
                output_attentions=False,
            )
            
            out_tensor = layer_outputs[0]
            new_past_key_values.append(layer_outputs[1])

        return out_tensor, new_past_key_values


    def forward(self, input_ids, past_key_values=None):
        """
        Forward pass through the model.
        """
        hidden_states, new_past_key_values = self.forward_layers(input_ids, past_key_values)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        if DEBUG >= 2:
            print(f"\nLogits shape: {logits.shape}")  # Debugging
        return logits, new_past_key_values
