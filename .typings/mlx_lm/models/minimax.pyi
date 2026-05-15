"""Type stubs for mlx_lm.models.minimax"""

from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs
from .switch_layers import SwitchGLU

@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    num_local_experts: int
    num_experts_per_tok: int
    max_position_embeddings: int

class MiniMaxAttention(nn.Module):
    num_heads: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    scale: float
    q_proj: nn.Linear
    k_proj: nn.Linear
    v_proj: nn.Linear
    o_proj: nn.Linear
    q_norm: nn.Module
    k_norm: nn.Module
    rope: nn.Module

    def __init__(self, args: ModelArgs) -> None: ...
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array: ...

class MiniMaxSparseMoeBlock(nn.Module):
    num_experts_per_tok: int
    gate: nn.Linear
    switch_mlp: SwitchGLU
    e_score_correction_bias: mx.array
    sharding_group: Optional[mx.distributed.Group]

    def __init__(self, args: ModelArgs) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class MiniMaxDecoderLayer(nn.Module):
    self_attn: MiniMaxAttention
    block_sparse_moe: MiniMaxSparseMoeBlock
    input_layernorm: nn.RMSNorm
    post_attention_layernorm: nn.RMSNorm

    def __init__(self, args: ModelArgs) -> None: ...
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array: ...

class MiniMaxModel(nn.Module):
    embed_tokens: nn.Embedding
    layers: list[MiniMaxDecoderLayer]
    norm: nn.RMSNorm

    def __init__(self, args: ModelArgs) -> None: ...
    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array: ...

class Model(nn.Module):
    model_type: str
    model: MiniMaxModel
    lm_head: nn.Linear

    def __init__(self, args: ModelArgs) -> None: ...
    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array: ...
    @property
    def layers(self) -> list[MiniMaxDecoderLayer]: ...
