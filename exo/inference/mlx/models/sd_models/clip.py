# Adapted from https://github.com/ml-explore/mlx-examples/blob/main/stable_diffusion/stable_diffusion/clip.py

import math
from dataclasses import dataclass
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
from dataclasses import field, dataclass
from exo.inference.shard import Shard
from exo.inference.mlx.models.base import IdentityBlock 

_ACTIVATIONS = {"quick_gelu": nn.gelu_fast_approx, "gelu": nn.gelu}



@dataclass
class CLIPTextModelConfig:
    num_layers: int = 23
    model_dims: int = 1024
    num_heads: int = 16
    max_length: int = 77
    vocab_size: int = 49408
    projection_dim: Optional[int] = None
    hidden_act: str = "quick_gelu"

    @classmethod
    def from_dict(cls, config):
        return ModelArgs(
            num_layers=config["num_hidden_layers"],
            model_dims=config["hidden_size"],
            num_heads=config["num_attention_heads"],
            max_length=config["max_position_embeddings"],
            vocab_size=config["vocab_size"],
            projection_dim=config["projection_dim"] if "WithProjection" in config['architectures'][0] else None,
            hidden_act=config.get("hidden_act", "quick_gelu"),
            weight_files=config.get("weight_files", [])
            )

@dataclass
class ModelArgs(CLIPTextModelConfig):
    shard: Shard = field(default_factory=lambda: Shard("", 0, 0, 0))
    weight_files: List[str] = field(default_factory=lambda: [])
    def __post_init__(self):
        if isinstance(self.shard, dict):
            self.shard = Shard(**self.shard)

        if not isinstance(self.shard, Shard):
            raise TypeError(f"Expected shard to be a Shard instance or a dict, got {type(self.shard)} instead")

        if not self.shard.is_first_layer():
            self.vision_config = None


@dataclass
class CLIPOutput:
    pooled_output: Optional[mx.array] = None
    last_hidden_state: Optional[mx.array] = None
    hidden_states: Optional[List[mx.array]] = None


class CLIPEncoderLayer(nn.Module):
    """The transformer encoder layer from CLIP."""

    def __init__(self, model_dims: int, num_heads: int, activation: str):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(model_dims)
        self.layer_norm2 = nn.LayerNorm(model_dims)

        self.attention = nn.MultiHeadAttention(model_dims, num_heads)
        self.attention.query_proj.bias = mx.zeros(model_dims)
        self.attention.key_proj.bias = mx.zeros(model_dims)
        self.attention.value_proj.bias = mx.zeros(model_dims)
        self.attention.out_proj.bias = mx.zeros(model_dims)

        self.linear1 = nn.Linear(model_dims, 4 * model_dims)
        self.linear2 = nn.Linear(4 * model_dims, model_dims)

        self.act = _ACTIVATIONS[activation]

    def __call__(self, x, attn_mask=None):
        
        y = self.layer_norm1(x)
        y = self.attention(y, y, y, attn_mask)
        x = y + x
        
        y = self.layer_norm2(x)
        y = self.linear1(y)
        y = self.act(y)
        y = self.linear2(y)
        x = y + x
        return x


class CLIPTextModel(nn.Module):
    """Implements the text encoder transformer from CLIP."""

    def __init__(self, config: CLIPTextModelConfig, shard: Shard):
        super().__init__()

        self.shard = shard
        self.layers_range = range(self.shard.start_layer*2, self.shard.end_layer*2+2) 
        if self.shard.is_first_layer():
            self.token_embedding = nn.Embedding(config.vocab_size, config.model_dims)
            self.position_embedding = nn.Embedding(config.max_length, config.model_dims)
        self.layers = []
        for i in range(math.ceil(config.num_layers/2)):
            if  2*i in self.layers_range:
                self.layers.append(CLIPEncoderLayer(config.model_dims, config.num_heads, config.hidden_act))
            if 2*i+1 in self.layers_range and 2*i+1 < config.num_layers:
                self.layers.append(CLIPEncoderLayer(config.model_dims, config.num_heads, config.hidden_act))
            else:
                self.layers.append(IdentityBlock())
        if self.shard.is_last_layer():
            self.final_layer_norm = nn.LayerNorm(config.model_dims)

        if config.projection_dim is not None:
            self.text_projection = nn.Linear(
                config.model_dims, config.projection_dim, bias=False
            )

    def _get_mask(self, N, dtype):
        indices = mx.arange(N)
        mask = indices[:, None] < indices[None]
        mask = mask.astype(dtype) * (-6e4 if dtype == mx.float16 else -1e9)
        return mask

    def __call__(self, x, mask=None):
        # Extract some shapes
        if self.shard.is_first_layer():
            B, N = x.shape
            eos_tokens = x.argmax(-1)
            
            # Compute the embeddings
            x = self.token_embedding(x)
           
            x = x + self.position_embedding.weight[:N]
            # Compute the features from the transformer
            mask = self._get_mask(N, x.dtype)
        
        for l in self.layers:
            x = l(x, mask)
        # Apply the final layernorm and return
        
        if self.shard.is_last_layer():
            x = self.final_layer_norm(x)
        
       

        return x, mask
    def sanitize(self, weights):
        sanitized_weights = {}
        for key, value in weights.items():
            if "position_ids" in key:
                continue
            if key.startswith("text_model."):
                key = key[11:]
            if key.startswith("embeddings."):
                key = key[11:]
            if key.startswith("encoder."):
                key = key[8:]

            # Map attention layers
            if "self_attn." in key:
                key = key.replace("self_attn.", "attention.")
            if "q_proj." in key:
                key = key.replace("q_proj.", "query_proj.")
            if "k_proj." in key:
                key = key.replace("k_proj.", "key_proj.")
            if "v_proj." in key:
                key = key.replace("v_proj.", "value_proj.")

            # Map ffn layers
            if "mlp.fc1" in key:
                key = key.replace("mlp.fc1", "linear1")
            if "mlp.fc2" in key:
                key = key.replace("mlp.fc2", "linear2")
            
            if key.startswith("layers."):
                layer_num = int(key.split(".")[1])
                if layer_num not in self.layers_range:
                    continue
            if not self.shard.is_first_layer() and "embedding" in key:
                continue
            if not self.shard.is_last_layer() and key.startswith("final_layer_norm"):
                continue
            if not self.shard.is_last_layer() and key.startswith("text_projection"):
                continue
            sanitized_weights[key] = value
        return sanitized_weights
