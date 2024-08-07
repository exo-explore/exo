# Work in progress on a generic hugging face model sharder
# right now doesn't work with all models

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from exo.inference.shard import Shard
import logging

class ShardedHuggingFaceModel(nn.Module):
    def __init__(self, model_name: str, shard: Shard):
        super(ShardedHuggingFaceModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.shard = shard
        self.device_ids = list(range(torch.cuda.device_count()))

        # Load the model
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            ))
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
        
        # Only keep layers corresponding to this shard
        self.layers = nn.ModuleList([
            self.model.transformer.h[i] for i in range(shard.start_layer, shard.end_layer + 1)
        ])

        logging.info(f"layers: {self.layers}")
        
        # Embeddings and final layer norm
        self.embed_tokens = self.full_model.model.embed_tokens
        self.embed_positions = self.full_model.model.embed_positions
        self.norm = self.full_model.model.norm
        self.lm_head = self.full_model.lm_head

    def forward_layers(self, input_ids, past_key_values=None):
        """
        Forward pass through the specified layers.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            past_key_values (list, optional): Past key values for caching.

        Returns:
            tuple: Hidden states and new past key values.
        """
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        # Token and position embeddings
        hidden_states = self.embed_tokens(input_ids) + self.embed_positions(input_ids)

        # Apply each layer in this shard
        new_past_key_values = []
        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i]
            hidden_states, new_layer_past = layer(hidden_states, past_key_values=layer_past, use_cache=True)
            new_past_key_values.append(new_layer_past)

        if self.shard.is_last_layer():
            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states)
            return logits, new_past_key_values
        else:
            return hidden_states, new_past_key_values