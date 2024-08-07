import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, DynamicCache
from exo.inference.shard import Shard

class ShardedHuggingFaceModel(nn.Module):
    def __init__(self, model_name: str, shard: Shard):
        super(ShardedHuggingFaceModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.shard = shard
        self.device_ids = list(range(torch.cuda.device_count()))

        # Load the model
        self.full_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        
        # Extract only the layers for this shard
        self.layers = nn.ModuleList([
            self.full_model.model.layers[i] for i in range(shard.start_layer, shard.end_layer + 1)
        ])

        # Embeddings and final layer norm
        self.embed_tokens = self.full_model.model.embed_tokens
        self.norm = self.full_model.model.norm
        self.lm_head = self.full_model.lm_head

    def forward_layers(self, input_ids, past_key_values=None):
        """
        Forward pass through the specified layers.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            past_key_values (DynamicCache, optional): Past key values for caching.

        Returns:
            tuple: Hidden states and new past key values.
        """
        if past_key_values is None:
            past_key_values = DynamicCache()

        # Token embeddings
        inputs_embeds = self.embed_tokens(input_ids)
        
        # Generate position ids
        position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # Apply each layer in this shard
        hidden_states = inputs_embeds
        new_past_key_values = DynamicCache()
        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if i < len(past_key_values) else None
            hidden_states, new_layer_past = layer(
                hidden_states,
                past_key_values=layer_past,
                use_cache=True,
                position_ids=position_ids
            )
            new_past_key_values.update(new_layer_past[0], new_layer_past[1], i)

        if self.shard.is_last_layer():
            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states)
            return logits, new_past_key_values
        else:
            return hidden_states, new_past_key_values
