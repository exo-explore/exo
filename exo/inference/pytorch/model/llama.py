import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from exo.inference.shard import Shard

class ShardedLLAMAModel(nn.Module):
    """
    Sharded LLAMA Model for performing inference with a subset of model layers.
    """

    def __init__(self, model_path: str, shard: Shard):
        """
        Initialize the ShardedLLAMAModel.

        Args:
            model_path (str): Path to the pretrained model.
            shard (Shard): Shard information indicating which layers to include.
        """
        super(ShardedLLAMAModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.shard = shard

        # Load the full model and move to device
        self.full_model = LlamaForCausalLM.from_pretrained(model_path)
        self.full_model.to(self.device)

        # Extract only the layers for this shard
        self.layers = nn.ModuleList([
            self.full_model.model.layers[i] for i in range(shard.start_layer, shard.end_layer + 1)
        ])

        # Embeddings and final layer norm
        self.embed_tokens = self.full_model.model.embed_tokens
        self.embed_positions = self.full_model.model.embed_positions
        self.norm = self.full_model.model.norm
        self.lm_head = self.full_model.lm_head

    def forward(self, input_ids, past_key_values=None):
        """
        Perform a forward pass through the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            past_key_values (list, optional): List of past key-value states for attention layers.

        Returns:
            tuple: Output logits or hidden states and the new past key-values.
        """
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        # Token and position embeddings
        hidden_states = self.embed_tokens(input_ids) + self.embed_positions(input_ids)

        # Apply each layer in this shard
        new_past_key_values = []
        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i]
            hidden_states, new_layer_past = layer(hidden_states, past_key_values=layer_past)
            new_past_key_values.append(new_layer_past)

        if self.shard.is_last_layer():
            # Apply final layer norm and compute logits
            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states)
            return logits, new_past_key_values
        else:
            return hidden_states, new_past_key_values
