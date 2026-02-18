from dataclasses import dataclass
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs
from .switch_layers import SwitchGLU

@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    max_position_embeddings: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    num_attention_heads: int
    n_group: int
    head_dim: int
    topk_group: int
    n_shared_experts: int
    n_routed_experts: int
    routed_scaling_factor: float
    num_experts_per_tok: int
    first_k_dense_replace: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: Optional[Dict[str, Any]]
    use_qk_norm: bool
    tie_word_embeddings: bool
    attention_bias: bool
    partial_rotary_factor: float
    scoring_func: str
    topk_method: str

class Attention(nn.Module):
    n_heads: int
    n_kv_heads: int
    scale: float
    q_proj: nn.Linear
    k_proj: nn.Linear
    v_proj: nn.Linear
    o_proj: nn.Linear
    use_qk_norm: bool
    q_norm: nn.RMSNorm
    k_norm: nn.RMSNorm
    rope: nn.RoPE

    def __init__(self, args: ModelArgs) -> None: ...
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array: ...

class MLP(nn.Module):
    config: ModelArgs
    hidden_size: int
    intermediate_size: int
    gate_proj: nn.Linear
    up_proj: nn.Linear
    down_proj: nn.Linear

    def __init__(
        self,
        config: ModelArgs,
        hidden_size: Optional[int] = None,
        intermediate_size: Optional[int] = None,
    ) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class MoEGate(nn.Module):
    config: ModelArgs
    top_k: int
    norm_topk_prob: bool
    n_routed_experts: int
    routed_scaling_factor: float
    n_group: int
    topk_group: int
    weight: mx.array
    e_score_correction_bias: mx.array

    def __init__(self, config: ModelArgs) -> None: ...
    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]: ...

class MoE(nn.Module):
    config: ModelArgs
    num_experts_per_tok: int
    switch_mlp: SwitchGLU
    gate: MoEGate
    shared_experts: MLP
    sharding_group: Optional[mx.distributed.Group]

    def __init__(self, config: ModelArgs) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class DecoderLayer(nn.Module):
    self_attn: Attention
    mlp: MLP | MoE
    input_layernorm: nn.RMSNorm
    post_attention_layernorm: nn.RMSNorm

    def __init__(self, config: ModelArgs, layer_idx: int) -> None: ...
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array: ...

class LanguageModel(nn.Module):
    vocab_size: int
    embed_tokens: nn.Embedding
    layers: list[DecoderLayer]
    norm: nn.RMSNorm
    pipeline_rank: int
    pipeline_size: int
    start_idx: int
    end_idx: Optional[int]
    num_layers: int

    def __init__(self, config: ModelArgs) -> None: ...
    def __call__(
        self,
        x: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array: ...
    @property
    def pipeline_layers(self) -> list[DecoderLayer]: ...

class Model(nn.Module):
    args: ModelArgs
    model_type: str
    model: LanguageModel
    lm_head: nn.Linear

    def __init__(self, config: ModelArgs) -> None: ...
    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array: ...
    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]: ...
    def shard(self, group: Optional[mx.distributed.Group] = None) -> None: ...
    @property
    def layers(self) -> list[DecoderLayer]: ...
    @property
    def cast_predicate(self) -> Any: ...
