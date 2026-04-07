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
    head_dim: int
    global_head_dim: int
    global_partial_rotary_factor: float
    rms_norm_eps: float
    vocab_size: int
    vocab_size_per_layer_input: int
    num_key_value_heads: int
    num_global_key_value_heads: Optional[int]
    num_kv_shared_layers: int
    pad_token_id: int
    hidden_size_per_layer_input: int
    rope_traditional: bool
    partial_rotary_factor: float
    rope_parameters: Optional[Dict[str, Any]]
    sliding_window: int
    sliding_window_pattern: int
    max_position_embeddings: int
    attention_k_eq_v: bool
    final_logit_softcapping: float
    use_double_wide_mlp: bool
    enable_moe_block: bool
    num_experts: Optional[int]
    top_k_experts: Optional[int]
    moe_intermediate_size: Optional[int]
    layer_types: Optional[List[str]]
    tie_word_embeddings: bool

    def __post_init__(self) -> None: ...

class MLP(nn.Module):
    gate_proj: nn.Linear
    down_proj: nn.Linear
    up_proj: nn.Linear

    def __init__(self, config: ModelArgs, layer_idx: int = 0) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class Router(nn.Module):
    proj: nn.Linear
    scale: mx.array
    per_expert_scale: mx.array

    def __init__(self, config: ModelArgs) -> None: ...
    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]: ...

class Experts(nn.Module):
    switch_glu: SwitchGLU

    def __init__(self, config: ModelArgs) -> None: ...
    def __call__(
        self, x: mx.array, top_k_indices: mx.array, top_k_weights: mx.array
    ) -> mx.array: ...

class Attention(nn.Module):
    layer_idx: int
    layer_type: str
    is_sliding: bool
    head_dim: int
    n_heads: int
    n_kv_heads: int
    use_k_eq_v: bool
    scale: float
    q_proj: nn.Linear
    k_proj: nn.Linear
    v_proj: nn.Linear
    o_proj: nn.Linear
    q_norm: nn.Module
    k_norm: nn.Module
    v_norm: nn.Module
    rope: nn.Module

    def __init__(self, config: ModelArgs, layer_idx: int) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

class DecoderLayer(nn.Module):
    layer_idx: int
    layer_type: str
    self_attn: Attention
    mlp: MLP
    enable_moe: bool
    router: Router
    experts: Experts
    input_layernorm: nn.Module
    post_attention_layernorm: nn.Module
    pre_feedforward_layernorm: nn.Module
    post_feedforward_layernorm: nn.Module
    post_feedforward_layernorm_1: nn.Module
    post_feedforward_layernorm_2: nn.Module
    pre_feedforward_layernorm_2: nn.Module
    hidden_size_per_layer_input: int
    per_layer_input_gate: Optional[nn.Linear]
    per_layer_projection: Optional[nn.Linear]
    post_per_layer_input_norm: Optional[nn.Module]
    layer_scalar: mx.array

    def __init__(self, config: ModelArgs, layer_idx: int) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

class Gemma4TextModel(nn.Module):
    config: ModelArgs
    vocab_size: int
    window_size: int
    sliding_window_pattern: int
    num_hidden_layers: int
    embed_tokens: nn.Embedding
    embed_scale: float
    layers: list[DecoderLayer]
    norm: nn.Module
    hidden_size_per_layer_input: int
    embed_tokens_per_layer: Optional[nn.Embedding]
    per_layer_model_projection: Optional[nn.Linear]
    per_layer_projection_norm: Optional[nn.Module]
    previous_kvs: list[int]

    def __init__(self, config: ModelArgs) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> mx.array: ...

class Model(nn.Module):
    args: ModelArgs
    model_type: str
    model: Gemma4TextModel
    final_logit_softcapping: float
    tie_word_embeddings: bool
    lm_head: nn.Linear

    def __init__(self, args: ModelArgs) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> mx.array: ...
    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]: ...
    @property
    def layers(self) -> list[DecoderLayer]: ...
    @property
    def head_dim(self) -> int: ...
    @property
    def n_kv_heads(self) -> int: ...
    @property
    def quant_predicate(self) -> Any: ...
    def make_cache(self) -> list[KVCache | RotatingKVCache]: ...
