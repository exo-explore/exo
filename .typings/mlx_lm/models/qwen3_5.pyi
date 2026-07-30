from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .cache import ArraysCache, KVCache
from .qwen3_next import (
    Qwen3NextAttention as Attention,
    Qwen3NextMLP as MLP,
    Qwen3NextRMSNormGated as RMSNormGated,
    Qwen3NextSparseMoeBlock,
)

SparseMoeBlock = Qwen3NextSparseMoeBlock
from .switch_layers import SwitchGLU

@dataclass
class TextModelArgs:
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    max_position_embeddings: int
    linear_num_value_heads: int
    linear_num_key_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    tie_word_embeddings: bool
    attention_bias: bool
    head_dim: Optional[int]
    full_attention_interval: int
    num_experts: int
    num_experts_per_tok: int
    decoder_sparse_step: int
    shared_expert_intermediate_size: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    rope_parameters: Optional[dict[str, Any]]
    partial_rotary_factor: float
    rope_theta: float
    rope_scaling: Optional[dict[str, Any]]

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> TextModelArgs: ...
    def __post_init__(self) -> None: ...

class GatedDeltaNet(nn.Module):
    hidden_size: int
    num_v_heads: int
    num_k_heads: int
    head_k_dim: int
    head_v_dim: int
    key_dim: int
    value_dim: int
    conv_kernel_size: int
    conv_dim: int
    conv1d: nn.Conv1d
    in_proj_qkv: nn.Linear
    in_proj_z: nn.Linear
    in_proj_b: nn.Linear
    in_proj_a: nn.Linear
    dt_bias: mx.array
    A_log: mx.array
    norm: RMSNormGated
    out_proj: nn.Linear

    def __init__(self, config: TextModelArgs) -> None: ...
    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array: ...

class DecoderLayer(nn.Module):
    is_linear: bool
    linear_attn: GatedDeltaNet
    self_attn: Attention
    input_layernorm: nn.RMSNorm
    post_attention_layernorm: nn.RMSNorm
    mlp: MLP | SparseMoeBlock

    def __init__(self, args: TextModelArgs, layer_idx: int) -> None: ...
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array: ...

class Qwen3_5TextModel(nn.Module):
    embed_tokens: nn.Embedding
    layers: list[DecoderLayer]
    norm: nn.RMSNorm
    ssm_idx: int
    fa_idx: int

    def __init__(self, args: TextModelArgs) -> None: ...
    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array: ...

class TextModel(nn.Module):
    args: TextModelArgs
    model_type: str
    model: Qwen3_5TextModel
    lm_head: nn.Linear

    def __init__(self, args: TextModelArgs) -> None: ...
    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array: ...
    @property
    def layers(self) -> list[DecoderLayer]: ...
    def make_cache(self) -> list[ArraysCache | KVCache]: ...
    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]: ...

@dataclass
class ModelArgs:
    model_type: str
    text_config: dict[str, Any]

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> ModelArgs: ...

class Model(nn.Module):
    args: ModelArgs
    model_type: str
    language_model: TextModel

    def __init__(self, args: ModelArgs) -> None: ...
    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array: ...
    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]: ...
    @property
    def layers(self) -> list[DecoderLayer]: ...
    def make_cache(self) -> list[ArraysCache | KVCache]: ...
