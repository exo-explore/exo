"""Type stubs for mlx_lm.models.deepseek_v3"""

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
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    n_shared_experts: Optional[int]
    n_routed_experts: Optional[int]
    routed_scaling_factor: float
    kv_lora_rank: int
    q_lora_rank: Optional[int]
    qk_rope_head_dim: int
    v_head_dim: int
    qk_nope_head_dim: int
    topk_method: str
    scoring_func: str
    norm_topk_prob: bool
    n_group: int
    topk_group: int
    num_experts_per_tok: int
    moe_layer_freq: int
    first_k_dense_replace: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: Optional[Dict[str, Any]]
    attention_bias: bool

class DeepseekV3Attention(nn.Module):
    config: ModelArgs
    hidden_size: int
    num_heads: int
    max_position_embeddings: int
    rope_theta: float
    q_lora_rank: Optional[int]
    qk_rope_head_dim: int
    kv_lora_rank: int
    v_head_dim: int
    qk_nope_head_dim: int
    q_head_dim: int
    scale: float
    q_proj: nn.Linear
    q_a_proj: nn.Linear
    q_a_layernorm: nn.RMSNorm
    q_b_proj: nn.Linear
    kv_a_proj_with_mqa: nn.Linear
    kv_a_layernorm: nn.RMSNorm
    kv_b_proj: nn.Linear
    o_proj: nn.Linear
    rope: Any

    def __init__(self, config: ModelArgs) -> None: ...
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array: ...

class DeepseekV3MLP(nn.Module):
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
    n_routed_experts: Optional[int]
    routed_scaling_factor: float
    n_group: int
    topk_group: int
    weight: mx.array
    e_score_correction_bias: mx.array

    def __init__(self, config: ModelArgs) -> None: ...
    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]: ...

class DeepseekV3MoE(nn.Module):
    config: ModelArgs
    num_experts_per_tok: int
    switch_mlp: SwitchGLU
    gate: MoEGate
    shared_experts: DeepseekV3MLP
    sharding_group: Optional[mx.distributed.Group]

    def __init__(self, config: ModelArgs) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class DeepseekV3DecoderLayer(nn.Module):
    self_attn: DeepseekV3Attention
    mlp: DeepseekV3MLP | DeepseekV3MoE
    input_layernorm: nn.RMSNorm
    post_attention_layernorm: nn.RMSNorm

    def __init__(self, config: ModelArgs, layer_idx: int) -> None: ...
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array: ...

class DeepseekV3Model(nn.Module):
    vocab_size: int
    embed_tokens: nn.Embedding
    layers: list[DeepseekV3DecoderLayer]
    norm: nn.RMSNorm

    def __init__(self, config: ModelArgs) -> None: ...
    def __call__(
        self,
        x: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array: ...

class Model(nn.Module):
    model_type: str
    model: DeepseekV3Model
    lm_head: nn.Linear

    def __init__(self, config: ModelArgs) -> None: ...
    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array: ...
    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]: ...
    @property
    def layers(self) -> list[DeepseekV3DecoderLayer]: ...
