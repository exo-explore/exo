import torch
import numpy as np
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from exo.inference.shard import Shard
from exo.helpers import DEBUG
from typing import Tuple

def sample_logits(logits, temp=0.85, top_k=25, top_p=0.9, alpha_f=0.1, alpha_p=0.0):
    # Apply temperature scaling
    if temp > 0:
        logits = logits / temp

    # Top-k sampling
    if top_k > 0:
        top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
        logits = torch.full_like(logits, -float('inf'))
        logits.scatter_(-1, top_k_indices, top_k_values)

    # Top-p (nucleus) sampling
    if 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, -float('inf'))

    # Alpha sampling (to discourage repetition)
    if alpha_f or alpha_p:
        if not hasattr(sample_logits, "alpha_counter"):
            setattr(sample_logits, "alpha_counter", torch.zeros_like(logits, dtype=torch.int32).contiguous())
        logits = logits - (sample_logits.alpha_counter * alpha_f + (sample_logits.alpha_counter > 0) * alpha_p)

    # Sample from the logits
    probabilities = F.softmax(logits, dim=-1)
    sampled_token = torch.multinomial(probabilities, 1)

    # Update alpha counter
    if alpha_f or alpha_p:
        condition = (torch.arange(probabilities.numel(), device=logits.device) == sampled_token)
        condition = condition.bool()  # Convert condition to boolean tensor
        sample_logits.alpha_counter = torch.where(condition, sample_logits.alpha_counter + 1, sample_logits.alpha_counter)

    return sampled_token

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
    ) -> np.ndarray:
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

        if self.shard.is_first_layer():
            hidden_states = self.embed_tokens(hidden_states)

            batch_size, seq_len = input_data.size()
            position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0).expand(batch_size, -1)

            position_embeddings = self.full_model.model.rotary_emb(
                hidden_states,
                position_ids
            )

            if DEBUG >= 2:
                print(f"embedded hidden_states {hidden_states}")
                print(f"position_ids: {position_embeddings}")

        for i, layer in enumerate(self.layers):
            # Forward pass through the layer
            if DEBUG >= 2:
                print(f"\n[layer {i}] {layer}")
                print(f"hidden_states {hidden_states}")
            
            layer_outputs = layer(
                hidden_states,
                position_embeddings=position_embeddings
            )

            if DEBUG >= 2:
                print(f"\n[layer {i}] layer_outputs: {layer_outputs[0]}")
            
            hidden_states = layer_outputs[0]

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

            return output_token
        
        return hidden_states.cpu().numpy()
