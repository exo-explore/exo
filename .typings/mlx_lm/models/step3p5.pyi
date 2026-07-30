from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs
from .switch_layers import SwitchGLU

@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    num_attention_heads: int
    num_attention_groups: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: Optional[Dict[str, Any]]
    max_position_embeddings: int
    sliding_window: int
    layer_types: Optional[List[str]]
    yarn_only_types: Optional[List[str]]
    partial_rotary_factors: Optional[List[float]]
    attention_other_setting: Optional[Dict[str, Any]]
    use_head_wise_attn_gate: bool
    moe_num_experts: int
    moe_top_k: int
    moe_intermediate_size: int
    share_expert_dim: int
    moe_layers_enum: Optional[str]
    moe_router_scaling_factor: float
    norm_expert_weight: bool
    swiglu_limits: Optional[List[float]]
    swiglu_limits_shared: Optional[List[float]]
    tie_word_embeddings: bool

class Step3p5MLP(nn.Module):
    hidden_size: int
    intermediate_size: int
    gate_proj: nn.Linear
    up_proj: nn.Linear
    down_proj: nn.Linear
    limit: Optional[float]

    def __init__(
        self, args: ModelArgs, intermediate_size: int, swiglu_limit: float = 0
    ) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class Step3p5MoEGate(nn.Module):
    top_k: int
    n_routed_experts: int
    routed_scaling_factor: float
    norm_topk_prob: bool
    gate: nn.Linear
    router_bias: mx.array

    def __init__(self, args: ModelArgs) -> None: ...
    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]: ...

class Step3p5MoE(nn.Module):
    gate: Step3p5MoEGate
    switch_mlp: SwitchGLU
    share_expert: Step3p5MLP
    sharding_group: Optional[mx.distributed.Group]

    def __init__(self, args: ModelArgs, layer_idx: int) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class Step3p5Attention(nn.Module):
    is_sliding: bool
    num_heads: int
    num_kv_heads: int
    head_dim: int
    scale: float
    q_proj: nn.Linear
    k_proj: nn.Linear
    v_proj: nn.Linear
    o_proj: nn.Linear
    q_norm: nn.Module
    k_norm: nn.Module
    use_head_wise_attn_gate: bool
    g_proj: nn.Linear
    rope: nn.Module

    def __init__(self, args: ModelArgs, layer_idx: int) -> None: ...
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array: ...

class Step3p5DecoderLayer(nn.Module):
    self_attn: Step3p5Attention
    is_sliding: bool
    is_moe_layer: bool
    mlp: Step3p5MLP | Step3p5MoE
    input_layernorm: nn.Module
    post_attention_layernorm: nn.Module

    def __init__(self, args: ModelArgs, layer_idx: int) -> None: ...
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array: ...

class Step3p5Model(nn.Module):
    args: ModelArgs
    vocab_size: int
    num_layers: int
    embed_tokens: nn.Embedding
    layers: list[Step3p5DecoderLayer]
    norm: nn.Module
    _swa_idx: Optional[int]
    _full_idx: Optional[int]

    def __init__(self, args: ModelArgs) -> None: ...
    def __call__(
        self,
        x: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array: ...

class Model(nn.Module):
    args: ModelArgs
    model_type: str
    model: Step3p5Model
    lm_head: nn.Linear

    def __init__(self, args: ModelArgs) -> None: ...
    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array: ...
    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]: ...
    def shard(self, group: Optional[mx.distributed.Group] = None) -> None: ...
    @property
    def layers(self) -> list[Step3p5DecoderLayer]: ...
    def make_cache(self) -> list[Any]: ...
    @property
    def cast_predicate(self) -> Any: ...
    @property
    def quant_predicate(self) -> Any: ...
