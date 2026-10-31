"""Type stubs for mlx_lm.models.flint"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs
from .cache import KVCache, RotatingKVCache
from .switch_layers import SwitchGLU

@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    mlp_only_layers: List[int]
    layer_types: List[str]
    sliding_window: int
    rms_norm_eps: float
    vocab_size: int
    rope_theta: float = ...
    rope_parameters: Optional[Dict[str, Any]] = ...
    tie_word_embeddings: bool = ...

class Attention(nn.Module):
    n_heads: int
    n_kv_heads: int
    scale: float
    q_proj: nn.Linear
    k_proj: nn.Linear
    v_proj: nn.Linear
    o_proj: nn.Linear
    q_norm: nn.RMSNorm
    k_norm: nn.RMSNorm
    softmax_offset: mx.array
    rope: Optional[nn.RoPE]

    def __init__(self, args: ModelArgs, layer_idx: int) -> None: ...
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array: ...

class MLP(nn.Module):
    gate_proj: nn.Linear
    down_proj: nn.Linear
    up_proj: nn.Linear

    def __init__(self, dim: int, hidden_dim: int) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class SparseMoeBlock(nn.Module):
    top_k: int
    gate: nn.Linear
    switch_mlp: SwitchGLU
    shared_expert: MLP
    sharding_group: Optional[mx.distributed.Group]
    _zero_bias: mx.array

    def __init__(self, args: ModelArgs) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class DecoderLayer(nn.Module):
    use_sliding: bool
    self_attn: Attention
    mlp: MLP | SparseMoeBlock
    input_layernorm: nn.RMSNorm
    post_attention_layernorm: nn.RMSNorm

    def __init__(self, args: ModelArgs, layer_idx: int) -> None: ...
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array: ...

class FlintModel(nn.Module):
    args: ModelArgs
    embed_tokens: nn.Embedding
    layers: list[DecoderLayer]
    norm: nn.RMSNorm
    fa_idx: Optional[int]
    swa_idx: Optional[int]

    def __init__(self, args: ModelArgs) -> None: ...
    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array: ...

class Model(nn.Module):
    args: ModelArgs
    model_type: str
    model: FlintModel
    lm_head: nn.Linear

    def __init__(self, args: ModelArgs) -> None: ...
    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array: ...
    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]: ...
    def shard(self, group: Optional[mx.distributed.Group] = None) -> None: ...
    @property
    def layers(self) -> list[DecoderLayer]: ...
    def make_cache(self) -> list[KVCache | RotatingKVCache]: ...
