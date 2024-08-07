import torch
import torch.nn as nn
from exo.inference.shard import Shard

class ShardedLLAMAModel(nn.Module):
    def __init__(self, model, shard: Shard):
        super(ShardedLLAMAModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.shard = shard

        # Only keep layers corresponding to this shard
        self.layers = nn.ModuleList([model.transformer.h[i] for i in range(shard.start_layer, shard.end_layer + 1)])

        # Move embeddings to the appropriate device
        self.model = model
        self.model.transformer.wte.to(self.device)
        self.model.transformer.wpe.to(self.device)

    def forward(self, input_ids, past_key_values=None):
        hidden_states = self._get_initial_hidden_states(input_ids)
        hidden_states, new_past_key_values = self._process_layers(hidden_states, past_key_values)

        if self.shard.is_last_layer():
            hidden_states = self.model.transformer.ln_f(hidden_states.to(self.device))
            logits = self.model.lm_head(hidden_states)
            return logits, new_past_key_values
        else:
            return hidden_states, new_past_key_values

    def _get_initial_hidden_states(self, input_ids):
        input_embeds = self.model.transformer.wte(input_ids.to(self.device))
        position_embeds = self.model.transformer.wpe(torch.arange(input_ids.shape[1], device=self.device))
        return input_embeds + position_embeds

    def _process_layers(self, hidden_states, past_key_values):
        new_past_key_values = []
        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values else None
            hidden_states, new_layer_past = layer(hidden_states, past_key_values=layer_past)
            new_past_key_values.append(new_layer_past)
        return hidden_states, new_past_key_values
