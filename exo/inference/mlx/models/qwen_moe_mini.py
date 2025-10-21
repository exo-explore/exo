"""
Qwen-MoE-Mini: A lightweight MoE model with pipeline support for 16GB Mac minis
Based on DeepSeek architecture patterns but optimized for small memory footprint
"""

from dataclasses import dataclass, field
from typing import Optional, List
import math

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.cache import KVCache
from .base import IdentityBlock
from exo.inference.shard import Shard


@dataclass
class ModelArgs:
    model_type: str = "qwen_moe_mini"
    vocab_size: int = 32000
    hidden_size: int = 1024  # Smaller than standard models
    intermediate_size: int = 2816  # Reduced for memory efficiency
    num_hidden_layers: int = 16  # Fewer layers for 16GB constraint
    num_attention_heads: int = 16
    num_key_value_heads: int = 4  # GQA for memory efficiency
    
    # MoE specific parameters
    n_routed_experts: int = 8  # Total number of experts
    n_shared_experts: int = 2  # Always-active experts
    num_experts_per_tok: int = 2  # Top-k routing
    moe_intermediate_size: int = 1408  # Expert FFN size
    shared_expert_intermediate_size: int = 2816
    
    # Standard parameters
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    max_position_embeddings: int = 8192
    tie_word_embeddings: bool = False
    
    # Sharding support
    shard: Shard = field(default_factory=lambda: Shard("", 0, 0, 0))
    
    def __post_init__(self):
        if isinstance(self.shard, Shard):
            return
        if not isinstance(self.shard, dict):
            raise TypeError(f"Expected shard to be a Shard instance or a dict, got {type(self.shard)}")
        self.shard = Shard(**self.shard)


class MoEGate(nn.Module):
    """Router for selecting experts"""
    def __init__(self, hidden_size: int, n_routed_experts: int, num_experts_per_tok: int):
        super().__init__()
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.gate = nn.Linear(hidden_size, n_routed_experts, bias=False)
    
    def __call__(self, x: mx.array) -> tuple:
        # x shape: [batch, seq_len, hidden_size]
        logits = self.gate(x)  # [batch, seq_len, n_experts]
        
        # Get top-k experts
        scores = mx.softmax(logits, axis=-1)
        expert_indices = mx.argpartition(-scores, kth=self.num_experts_per_tok-1, axis=-1)
        expert_indices = expert_indices[..., :self.num_experts_per_tok]
        
        # Get weights for selected experts
        batch_size, seq_len = x.shape[:2]
        expert_weights = mx.take_along_axis(scores, expert_indices, axis=-1)
        expert_weights = expert_weights / mx.sum(expert_weights, axis=-1, keepdims=True)
        
        return expert_indices, expert_weights


class MoEBlock(nn.Module):
    """Mixture of Experts FFN block"""
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        
        # Router
        self.gate = MoEGate(
            config.hidden_size,
            config.n_routed_experts,
            config.num_experts_per_tok
        )
        
        # Shared experts (always active)
        self.shared_experts = nn.Sequential(
            nn.Linear(config.hidden_size, config.shared_expert_intermediate_size, bias=False),
            nn.SiLU(),
            nn.Linear(config.shared_expert_intermediate_size, config.hidden_size, bias=False)
        )
        
        # Routed experts
        self.experts = []
        for _ in range(config.n_routed_experts):
            expert = nn.Sequential(
                nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False),
                nn.SiLU(),
                nn.Linear(config.moe_intermediate_size, config.hidden_size, bias=False)
            )
            self.experts.append(expert)
    
    def __call__(self, x: mx.array) -> mx.array:
        batch_size, seq_len, hidden_size = x.shape
        
        # Shared experts contribution (always active)
        shared_output = self.shared_experts(x)
        
        # Get routing decisions
        expert_indices, expert_weights = self.gate(x)
        
        # Process through selected experts
        expert_output = mx.zeros_like(x)
        for i in range(self.num_experts_per_tok):
            for expert_id in range(self.n_routed_experts):
                # Create mask for tokens routed to this expert
                mask = (expert_indices[..., i] == expert_id)
                if mx.any(mask):
                    # Get tokens for this expert
                    expert_input = x * mask[..., None]
                    # Process through expert
                    output = self.experts[expert_id](expert_input)
                    # Weight and accumulate
                    weight = expert_weights[..., i:i+1] * mask[..., None]
                    expert_output = expert_output + output * weight
        
        # Combine shared and routed expert outputs
        return shared_output + expert_output


class QwenMoEMiniDecoderLayer(nn.Module):
    """Transformer layer with MoE FFN"""
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Self-attention
        self.self_attn = nn.MultiHeadAttention(
            dims=config.hidden_size,
            num_heads=config.num_attention_heads,
            bias=False
        )
        
        # MoE FFN
        self.mlp = MoEBlock(config)
        
        # Layer norms
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None
    ) -> mx.array:
        # Self-attention with residual
        residual = x
        x = self.input_layernorm(x)
        # MultiHeadAttention doesn't take cache directly
        x = self.self_attn(x, x, x, mask=mask)
        x = residual + x
        
        # MoE FFN with residual
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


class QwenMoEMiniModel(nn.Module):
    """Main model class with pipeline parallelism support"""
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.num_hidden_layers = config.num_hidden_layers
        self.vocab_size = config.vocab_size
        
        # Embeddings (only on first shard)
        if self.args.shard.is_first_layer():
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer layers
        self.layers = []
        for i in range(self.num_hidden_layers):
            if self.args.shard.start_layer <= i <= self.args.shard.end_layer:
                self.layers.append(QwenMoEMiniDecoderLayer(config))
            else:
                self.layers.append(IdentityBlock())
        
        # Final norm (only on last shard)
        if self.args.shard.is_last_layer():
            self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def __call__(
        self,
        x: mx.array,
        cache: Optional[KVCache] = None
    ) -> mx.array:
        # Embedding
        if self.args.shard.is_first_layer():
            h = self.embed_tokens(x)
        else:
            h = x
        
        # Create causal mask if needed
        mask = None
        T = h.shape[1]
        if T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
            mask = mask.astype(h.dtype)
        
        # Process through layers
        if cache is None:
            cache = [None] * len(self.layers)
        
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
        
        # Final norm
        if self.args.shard.is_last_layer():
            h = self.norm(h)
        
        return h


class Model(nn.Module):
    """Top-level model with LM head"""
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = QwenMoEMiniModel(config)
        
        # Language modeling head (only on last shard)
        if self.args.shard.is_last_layer():
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[KVCache] = None
    ):
        out = self.model(inputs, cache)
        if self.args.shard.is_last_layer():
            return self.lm_head(out)
        return out
    
    def sanitize(self, weights):
        """Clean and prepare weights for sharded execution"""
        shard_state_dict = {}
        
        for key, value in weights.items():
            # Handle layer weights
            if key.startswith('model.layers.'):
                layer_num = int(key.split('.')[2])
                if self.args.shard.start_layer <= layer_num <= self.args.shard.end_layer:
                    shard_state_dict[key] = value
            # Handle embeddings
            elif self.args.shard.is_first_layer() and key.startswith('model.embed_tokens'):
                shard_state_dict[key] = value
            # Handle final norm and LM head
            elif self.args.shard.is_last_layer() and (key.startswith('model.norm') or key.startswith('lm_head')):
                shard_state_dict[key] = value
        
        # Handle expert weight stacking if needed
        for l in range(self.args.num_hidden_layers):
            if self.args.shard.start_layer <= l <= self.args.shard.end_layer:
                prefix = f"model.layers.{l}.mlp"
                
                # Stack individual expert weights into single tensor
                for expert_type in ["experts"]:
                    for param in ["weight"]:
                        # Check if we have separate expert weights
                        expert_keys = [f"{prefix}.{expert_type}.{i}.0.{param}" for i in range(self.args.n_routed_experts)]
                        if all(k in shard_state_dict for k in expert_keys):
                            # Stack them
                            stacked = mx.stack([shard_state_dict.pop(k) for k in expert_keys])
                            shard_state_dict[f"{prefix}.{expert_type}.{param}"] = stacked
        
        return shard_state_dict
    
    @property
    def layers(self):
        return self.model.layers
    
    @property 
    def head_dim(self):
        return self.args.hidden_size // self.args.num_attention_heads
    
    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads