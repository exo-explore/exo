import torch
from transformers import AutoModelForCausalLM
from exo.inference.shard import Shard
from exo.helpers import DEBUG
from typing import Tuple

class ShardedHuggingFaceModel(torch.nn.Module):
    def __init__(self, shard: Shard):
        super(ShardedHuggingFaceModel, self).__init__()

        if DEBUG >= 2:
            print(f"ShardedHuggingFaceModel init with shard {shard}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.shard = shard

        # Load the model
        self.full_model = AutoModelForCausalLM.from_pretrained(
            shard.model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        
        # Extract only the layers for this shard
        self.layers = torch.nn.ModuleList([
            self.full_model.model.layers[i] for i in range(shard.start_layer, shard.end_layer + 1)
        ])

        # Embeddings and final layer norm
        self.embed_tokens = self.full_model.model.embed_tokens
        self.norm = self.full_model.model.norm
        self.lm_head = self.full_model.lm_head

    def prefill(self, tokens: torch.tensor, start_pos: int=0) -> int:
        """
        Process the initial input tokens and set up the initial hidden states.
        """
         # Assuming tokens is a 1D tensor of token IDs
        for token in tokens:
            # Convert token to a tensor and get embeddings
            token_tensor = torch.tensor([[token]], device=self.device)
            inputs_embeds = self.embed_tokens(token_tensor)

            # Prefill with tokens
            for layer in self.layers:
                _ = layer(
                    inputs_embeds,
                    use_cache=True,
                    output_attentions=False,
                )
                # Update embeddings with layer output
                inputs_embeds = layer_outputs[0]

            # Increment start position
            start_pos += 1
        
        return start_pos

    def forward_layers(self, input_ids, past_key_values=None):
        """
        Forward pass through the specified layers.
        """
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Generate position ids
        seq_length = input_ids.shape[1]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(input_ids.shape)

        # Apply each layer in this shard
        new_past_key_values = []
        for i, layer in enumerate(self.layers):
            layer_outputs = layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                past_key_value=past_key_values[i],
                use_cache=True,
                output_attentions=False,
            )
            hidden_states = layer_outputs[0]
            new_past_key_values.append(layer_outputs[1])

        if self.shard.is_last_layer():
            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states)
            return logits, new_past_key_values
        else:
            return hidden_states, new_past_key_values

    def is_last_layer(self):
        return self.shard.is_last_layer()